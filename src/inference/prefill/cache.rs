use super::super::{NativeMlxSession, env_flag};
use super::config::{prefix_cache_max_bytes, prefix_cache_max_entries, prefix_cache_max_tokens};

#[derive(Clone, Debug)]
pub(in crate::inference) struct PromptStateCache {
    pub(super) prompt_ids: Vec<u32>,
    pub(super) state: crate::qwen36::DecodeState,
    pub(super) logits: mlx_rs::Array,
    pub(super) hidden: mlx_rs::Array,
    pub(super) mtp_history_state: crate::qwen36::MtpDecodeState,
    pub(super) estimated_bytes: usize,
}

pub(super) fn best_prompt_cache_match_index(
    session: &NativeMlxSession,
    prompt_ids: &[u32],
) -> Option<usize> {
    session
        .prompt_caches
        .iter()
        .enumerate()
        .filter(|(_, cached)| is_token_prefix(&cached.prompt_ids, prompt_ids))
        .max_by_key(|(_, cached)| cached.prompt_ids.len())
        .map(|(index, _)| index)
}

fn is_token_prefix(prefix: &[u32], full: &[u32]) -> bool {
    prefix.len() <= full.len() && full.starts_with(prefix)
}

pub(super) fn maybe_store_prompt_cache(
    session: &mut NativeMlxSession,
    prompt_ids: &[u32],
    state: &crate::qwen36::DecodeState,
    logits: &mlx_rs::Array,
    hidden: &mlx_rs::Array,
    mtp_history_state: &crate::qwen36::MtpDecodeState,
) {
    if !env_flag("MTPLX_PREFIX_CACHE", false) {
        session.prompt_caches.clear();
        return;
    }
    let max_tokens = prefix_cache_max_tokens();
    let max_entries = prefix_cache_max_entries();
    if max_entries == 0 {
        session.prompt_caches.clear();
        return;
    }
    if max_tokens == 0 || prompt_ids.len() > max_tokens {
        return;
    }
    if let Some(index) = session
        .prompt_caches
        .iter()
        .position(|cached| cached.prompt_ids == prompt_ids)
    {
        session.prompt_caches.remove(index);
    }
    session.prompt_caches.push(PromptStateCache {
        prompt_ids: prompt_ids.to_vec(),
        state: state.clone(),
        logits: logits.clone(),
        hidden: hidden.clone(),
        mtp_history_state: mtp_history_state.clone(),
        estimated_bytes: estimate_prompt_cache_bytes(
            prompt_ids,
            state,
            logits,
            hidden,
            mtp_history_state,
        ),
    });
    evict_prompt_caches(session, max_entries, prefix_cache_max_bytes());
}

fn evict_prompt_caches(session: &mut NativeMlxSession, max_entries: usize, max_bytes: usize) {
    while session.prompt_caches.len() > max_entries {
        session.prompt_caches.remove(0);
    }
    if max_bytes == 0 {
        session.prompt_caches.clear();
        return;
    }
    while prompt_cache_total_bytes(&session.prompt_caches) > max_bytes {
        let Some((index, _)) = session
            .prompt_caches
            .iter()
            .enumerate()
            .min_by_key(|(_, cache)| cache.prompt_ids.len())
        else {
            break;
        };
        session.prompt_caches.remove(index);
    }
}

fn prompt_cache_total_bytes(caches: &[PromptStateCache]) -> usize {
    caches
        .iter()
        .map(|cache| cache.estimated_bytes)
        .fold(0_usize, usize::saturating_add)
}

fn estimate_prompt_cache_bytes(
    prompt_ids: &[u32],
    state: &crate::qwen36::DecodeState,
    logits: &mlx_rs::Array,
    hidden: &mlx_rs::Array,
    mtp_history_state: &crate::qwen36::MtpDecodeState,
) -> usize {
    prompt_ids
        .len()
        .saturating_mul(std::mem::size_of::<u32>())
        .saturating_add(estimate_decode_state_bytes(state))
        .saturating_add(estimate_array_bytes(logits))
        .saturating_add(estimate_array_bytes(hidden))
        .saturating_add(estimate_full_attention_cache_bytes(
            &mtp_history_state.cache,
        ))
}

fn estimate_decode_state_bytes(state: &crate::qwen36::DecodeState) -> usize {
    state
        .layers
        .iter()
        .map(|layer| match layer {
            crate::qwen36::LayerDecodeState::Full(cache) => {
                estimate_full_attention_cache_bytes(cache)
            }
            crate::qwen36::LayerDecodeState::Linear(cache) => estimate_linear_cache_bytes(cache),
        })
        .fold(0_usize, usize::saturating_add)
}

fn estimate_full_attention_cache_bytes(cache: &crate::qwen36::FullAttentionCache) -> usize {
    cache
        .k
        .as_ref()
        .map(estimate_array_bytes)
        .unwrap_or(0)
        .saturating_add(cache.v.as_ref().map(estimate_array_bytes).unwrap_or(0))
}

fn estimate_linear_cache_bytes(cache: &crate::qwen36::LinearAttentionCache) -> usize {
    cache
        .conv_state
        .len()
        .saturating_add(cache.recurrent_state.len())
        .saturating_add(cache.conv_out.len())
        .saturating_add(cache.recurrent_out.len())
        .saturating_add(cache.q_normed.len())
        .saturating_add(cache.k_normed.len())
        .saturating_mul(std::mem::size_of::<f32>())
        .saturating_add(
            cache
                .metal_conv_state
                .as_ref()
                .map(estimate_array_bytes)
                .unwrap_or(0),
        )
        .saturating_add(
            cache
                .metal_recurrent_state
                .as_ref()
                .map(estimate_array_bytes)
                .unwrap_or(0),
        )
        .saturating_add(
            cache
                .metal_conv_block_states
                .as_ref()
                .map(estimate_array_bytes)
                .unwrap_or(0),
        )
        .saturating_add(
            cache
                .metal_recurrent_block_states
                .as_ref()
                .map(estimate_array_bytes)
                .unwrap_or(0),
        )
}

fn estimate_array_bytes(array: &mlx_rs::Array) -> usize {
    array
        .shape()
        .iter()
        .try_fold(1_usize, |product, dim| {
            usize::try_from(*dim)
                .ok()
                .and_then(|dim| product.checked_mul(dim))
        })
        .unwrap_or(0)
        .saturating_mul(dtype_bytes(array.dtype()))
}

fn dtype_bytes(dtype: mlx_rs::Dtype) -> usize {
    match dtype {
        mlx_rs::Dtype::Bool | mlx_rs::Dtype::Uint8 | mlx_rs::Dtype::Int8 => 1,
        mlx_rs::Dtype::Uint16
        | mlx_rs::Dtype::Int16
        | mlx_rs::Dtype::Float16
        | mlx_rs::Dtype::Bfloat16 => 2,
        mlx_rs::Dtype::Uint32 | mlx_rs::Dtype::Int32 | mlx_rs::Dtype::Float32 => 4,
        mlx_rs::Dtype::Uint64
        | mlx_rs::Dtype::Int64
        | mlx_rs::Dtype::Float64
        | mlx_rs::Dtype::Complex64 => 8,
    }
}
