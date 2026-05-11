use std::time::Instant;

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;

use super::{
    InferenceRequest, NativeMlxSession, PromptMessage, check_kill_switch, clear_mlx_cache,
    env_flag, log_memory_sample, memory_trace_enabled, mtp::append_mtp_history_from_commit,
};

mod cache;
mod config;

pub(super) use cache::PromptStateCache;
use cache::{best_prompt_cache_match_index, maybe_store_prompt_cache};
pub(super) use config::post_generation_cache_enabled;
use config::{
    prefill_chunk_tokens, prefill_eval_interval_chunks, prefix_cache_max_tokens,
    should_eval_prefill_chunk,
};

pub(super) struct PreparedPrompt {
    pub(super) state: crate::qwen36::DecodeState,
    pub(super) logits: mlx_rs::Array,
    pub(super) hidden: mlx_rs::Array,
    pub(super) mtp_history_state: crate::qwen36::MtpDecodeState,
    pub(super) prefill_s: f64,
    pub(super) prefix_cache_hit: bool,
    pub(super) prefix_cache_tokens: usize,
}

fn seed_mtp_history_from_prompt(
    session: &NativeMlxSession,
    prompt_ids: &[u32],
    prompt_hidden: &mlx_rs::Array,
) -> Result<crate::qwen36::MtpDecodeState> {
    let mut state = session.qwen.new_mtp_decode_state()?;
    for index in 1..prompt_ids.len() {
        let row = index as i32 - 1;
        let source_hidden = prompt_hidden.index((.., row..row + 1, ..));
        let (_logits, _hidden) = session.qwen.mtp_draft_step(
            prompt_ids[index],
            &source_hidden,
            &session.plan,
            &mut state,
        )?;
    }
    Ok(state)
}

pub(super) fn prepare_prompt_state(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    prompt_ids: &[u32],
) -> Result<PreparedPrompt> {
    let started = Instant::now();
    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
    let needs_mtp_history =
        persistent_mtp && request.mtp && request.depth > 0 && session.qwen.mtp.is_some();

    if env_flag("MTPLX_PREFIX_CACHE", false) {
        if let Some(index) = best_prompt_cache_match_index(session, prompt_ids) {
            let cached = session.prompt_caches.remove(index);
            let cached_prompt_len = cached.prompt_ids.len();
            let suffix = &prompt_ids[cached_prompt_len..];
            if !needs_mtp_history && !suffix.is_empty() {
                session.prompt_caches.push(cached);
                let (state, logits, hidden, mtp_history_state) =
                    prepare_prompt_state_chunked(session, request, prompt_ids, needs_mtp_history)?;
                let prefill_s = started.elapsed().as_secs_f64();
                maybe_store_prompt_cache(
                    session,
                    prompt_ids,
                    &state,
                    &logits,
                    &hidden,
                    &mtp_history_state,
                );
                return Ok(PreparedPrompt {
                    state,
                    logits,
                    hidden,
                    mtp_history_state,
                    prefill_s,
                    prefix_cache_hit: false,
                    prefix_cache_tokens: 0,
                });
            }
            let mut state = cached.state;
            let mut logits = cached.logits;
            let mut hidden = cached.hidden;
            let mut mtp_history_state = if needs_mtp_history {
                cached.mtp_history_state
            } else {
                session.qwen.new_mtp_decode_state().unwrap_or_default()
            };
            if !suffix.is_empty() {
                extend_prompt_state_chunked(
                    session,
                    request,
                    suffix,
                    &mut state,
                    &mut logits,
                    &mut hidden,
                    &mut mtp_history_state,
                    needs_mtp_history,
                )?;
            }
            state.clear_transient_block_states();
            eval_prepared_prompt_state(&state, &logits, &hidden, &mtp_history_state)?;
            let prefill_s = started.elapsed().as_secs_f64();
            maybe_store_prompt_cache(
                session,
                prompt_ids,
                &state,
                &logits,
                &hidden,
                &mtp_history_state,
            );
            return Ok(PreparedPrompt {
                state,
                logits,
                hidden,
                mtp_history_state,
                prefill_s,
                prefix_cache_hit: true,
                prefix_cache_tokens: cached_prompt_len,
            });
        }
    }

    let (state, logits, hidden, mtp_history_state) =
        prepare_prompt_state_chunked(session, request, prompt_ids, needs_mtp_history)?;
    let prefill_s = started.elapsed().as_secs_f64();
    maybe_store_prompt_cache(
        session,
        prompt_ids,
        &state,
        &logits,
        &hidden,
        &mtp_history_state,
    );
    Ok(PreparedPrompt {
        state,
        logits,
        hidden,
        mtp_history_state,
        prefill_s,
        prefix_cache_hit: false,
        prefix_cache_tokens: 0,
    })
}

fn prepare_prompt_state_chunked(
    session: &NativeMlxSession,
    request: &InferenceRequest,
    prompt_ids: &[u32],
    needs_mtp_history: bool,
) -> Result<(
    crate::qwen36::DecodeState,
    mlx_rs::Array,
    mlx_rs::Array,
    crate::qwen36::MtpDecodeState,
)> {
    let _ = request;
    if prompt_ids.is_empty() {
        anyhow::bail!("prompt token list cannot be empty");
    }

    let mut state = session.qwen.new_decode_state(&session.plan)?;
    let mut logits: Option<mlx_rs::Array> = None;
    let mut hidden: Option<mlx_rs::Array> = None;
    let mut mtp_history_state = if needs_mtp_history {
        session.qwen.new_mtp_decode_state()?
    } else {
        session.qwen.new_mtp_decode_state().unwrap_or_default()
    };

    if !needs_mtp_history {
        check_kill_switch("prefill_mlx_before")?;
        let input = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, input.len() as i32]);
        let (logits, hidden, _hidden_sequence) = session
            .qwen
            .prefill_decode_state_with_hidden_sequence(&input_ids, &session.plan, &mut state)?;
        state.clear_transient_block_states();
        eval_prepared_prompt_state(&state, &logits, &hidden, &mtp_history_state)?;

        if memory_trace_enabled() {
            log_memory_sample(
                "infer",
                "prefill_mlx",
                &format!("tokens={} position={}", prompt_ids.len(), state.position),
            );
        }
        check_kill_switch("prefill_mlx_after")?;
        return Ok((state, logits, hidden, mtp_history_state));
    }

    let chunk_tokens = prefill_chunk_tokens(prompt_ids.len());
    let eval_interval = prefill_eval_interval_chunks();
    let mut consumed_tokens = 0_usize;
    for (chunk_index, chunk) in prompt_ids.chunks(chunk_tokens).enumerate() {
        check_kill_switch("prefill_chunk_before")?;
        let is_last_chunk = consumed_tokens + chunk.len() == prompt_ids.len();
        consumed_tokens += chunk.len();

        if !needs_mtp_history && !is_last_chunk {
            session
                .qwen
                .decode_tokens_cache_only(chunk, &session.plan, &mut state)?;
            state.clear_transient_block_states();
            if should_eval_prefill_chunk(chunk_index, eval_interval) {
                eval_decode_state(&state)?;
                clear_mlx_cache();
            }

            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "prefill_chunk_cache_only",
                    &format!(
                        "chunk_index={chunk_index} chunk_tokens={} position={}",
                        chunk.len(),
                        state.position
                    ),
                );
            }
            check_kill_switch("prefill_chunk_after")?;
            continue;
        }

        let previous_hidden = hidden.clone();
        let chunk_hidden = if needs_mtp_history {
            let chunk_hidden =
                session
                    .qwen
                    .decode_tokens_hidden(chunk, &session.plan, &mut state)?;
            chunk_hidden.eval()?;
            Some(chunk_hidden)
        } else {
            None
        };
        let last_hidden = if let Some(chunk_hidden) = &chunk_hidden {
            let last = chunk.len() as i32 - 1;
            chunk_hidden.index((.., last..last + 1, ..))
        } else {
            session
                .qwen
                .decode_tokens_last_hidden(chunk, &session.plan, &mut state)?
        };
        state.clear_transient_block_states();

        if needs_mtp_history {
            if let Some(previous_hidden) = previous_hidden {
                append_mtp_history_from_commit(
                    session,
                    &mut mtp_history_state,
                    &previous_hidden,
                    chunk,
                    chunk_hidden.as_ref().expect("MTP chunk hidden exists"),
                )?;
            } else {
                mtp_history_state = seed_mtp_history_from_prompt(
                    session,
                    chunk,
                    chunk_hidden.as_ref().expect("MTP chunk hidden exists"),
                )?;
            }
            eval_mtp_decode_state(&mtp_history_state)?;
        }

        let last_logits = session.qwen.logits_from_hidden(&last_hidden)?;
        last_hidden.eval()?;
        last_logits.eval()?;
        eval_decode_state(&state)?;
        clear_mlx_cache();

        if memory_trace_enabled() {
            log_memory_sample(
                "infer",
                "prefill_chunk",
                &format!(
                    "chunk_index={chunk_index} chunk_tokens={} position={}",
                    chunk.len(),
                    state.position
                ),
            );
        }
        check_kill_switch("prefill_chunk_after")?;

        logits = Some(last_logits);
        hidden = Some(last_hidden);
    }

    let logits = logits.expect("non-empty prompt produces logits");
    let hidden = hidden.expect("non-empty prompt produces hidden state");
    state.clear_transient_block_states();
    eval_prepared_prompt_state(&state, &logits, &hidden, &mtp_history_state)?;
    Ok((state, logits, hidden, mtp_history_state))
}

fn extend_prompt_state_chunked(
    session: &NativeMlxSession,
    request: &InferenceRequest,
    suffix: &[u32],
    state: &mut crate::qwen36::DecodeState,
    logits: &mut mlx_rs::Array,
    hidden: &mut mlx_rs::Array,
    mtp_history_state: &mut crate::qwen36::MtpDecodeState,
    needs_mtp_history: bool,
) -> Result<()> {
    let _ = request;
    if suffix.is_empty() {
        return Ok(());
    }
    let eval_interval = prefill_eval_interval_chunks();
    let mut consumed_tokens = 0_usize;
    let total_prompt_tokens = state.position.max(0) as usize + suffix.len();
    let chunk_tokens = prefill_chunk_tokens(total_prompt_tokens);
    if !needs_mtp_history {
        let body_len = suffix.len().saturating_sub(1);
        let body = &suffix[..body_len];
        for (chunk_index, chunk) in body.chunks(chunk_tokens).enumerate() {
            check_kill_switch("prefill_suffix_chunk_before")?;
            session
                .qwen
                .decode_tokens_cache_only(chunk, &session.plan, state)?;
            state.clear_transient_block_states();
            if should_eval_prefill_chunk(chunk_index, eval_interval) {
                eval_decode_state(state)?;
                clear_mlx_cache();
            }

            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "prefill_suffix_chunk_cache_only",
                    &format!(
                        "chunk_index={chunk_index} chunk_tokens={} position={}",
                        chunk.len(),
                        state.position
                    ),
                );
            }
            check_kill_switch("prefill_suffix_chunk_after")?;
        }

        let final_chunk = &suffix[body_len..];
        let last_hidden =
            session
                .qwen
                .decode_tokens_last_hidden(final_chunk, &session.plan, state)?;
        state.clear_transient_block_states();
        let last_logits = session.qwen.logits_from_hidden(&last_hidden)?;
        last_hidden.eval()?;
        last_logits.eval()?;
        eval_decode_state(state)?;
        clear_mlx_cache();

        if memory_trace_enabled() {
            log_memory_sample(
                "infer",
                "prefill_suffix_chunk",
                &format!(
                    "chunk_index={} chunk_tokens={} position={}",
                    body.chunks(chunk_tokens).count(),
                    final_chunk.len(),
                    state.position
                ),
            );
        }
        check_kill_switch("prefill_suffix_chunk_after")?;

        *logits = last_logits;
        *hidden = last_hidden;
        return Ok(());
    }

    for (chunk_index, chunk) in suffix.chunks(chunk_tokens).enumerate() {
        check_kill_switch("prefill_suffix_chunk_before")?;
        let is_last_chunk = consumed_tokens + chunk.len() == suffix.len();
        consumed_tokens += chunk.len();

        if !needs_mtp_history && !is_last_chunk {
            session
                .qwen
                .decode_tokens_cache_only(chunk, &session.plan, state)?;
            state.clear_transient_block_states();
            if should_eval_prefill_chunk(chunk_index, eval_interval) {
                eval_decode_state(state)?;
                clear_mlx_cache();
            }

            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "prefill_suffix_chunk_cache_only",
                    &format!(
                        "chunk_index={chunk_index} chunk_tokens={} position={}",
                        chunk.len(),
                        state.position
                    ),
                );
            }
            check_kill_switch("prefill_suffix_chunk_after")?;
            continue;
        }

        let previous_hidden = hidden.clone();
        let chunk_hidden = if needs_mtp_history {
            let chunk_hidden = session
                .qwen
                .decode_tokens_hidden(chunk, &session.plan, state)?;
            chunk_hidden.eval()?;
            Some(chunk_hidden)
        } else {
            None
        };
        let last_hidden = if let Some(chunk_hidden) = &chunk_hidden {
            let last = chunk.len() as i32 - 1;
            chunk_hidden.index((.., last..last + 1, ..))
        } else {
            session
                .qwen
                .decode_tokens_last_hidden(chunk, &session.plan, state)?
        };
        state.clear_transient_block_states();

        if needs_mtp_history {
            append_mtp_history_from_commit(
                session,
                mtp_history_state,
                &previous_hidden,
                chunk,
                chunk_hidden.as_ref().expect("MTP chunk hidden exists"),
            )?;
            eval_mtp_decode_state(mtp_history_state)?;
        }

        let last_logits = session.qwen.logits_from_hidden(&last_hidden)?;
        last_hidden.eval()?;
        last_logits.eval()?;
        eval_decode_state(state)?;
        clear_mlx_cache();

        if memory_trace_enabled() {
            log_memory_sample(
                "infer",
                "prefill_suffix_chunk",
                &format!(
                    "chunk_index={chunk_index} chunk_tokens={} position={}",
                    chunk.len(),
                    state.position
                ),
            );
        }
        check_kill_switch("prefill_suffix_chunk_after")?;

        *logits = last_logits;
        *hidden = last_hidden;
    }
    Ok(())
}

fn eval_prepared_prompt_state(
    state: &crate::qwen36::DecodeState,
    logits: &mlx_rs::Array,
    hidden: &mlx_rs::Array,
    mtp_history_state: &crate::qwen36::MtpDecodeState,
) -> Result<()> {
    logits.eval()?;
    hidden.eval()?;
    eval_decode_state(state)?;
    eval_mtp_decode_state(mtp_history_state)?;
    clear_mlx_cache();
    Ok(())
}

fn eval_decode_state(state: &crate::qwen36::DecodeState) -> Result<()> {
    for layer in &state.layers {
        match layer {
            crate::qwen36::LayerDecodeState::Full(cache) => {
                eval_full_attention_cache(cache)?;
            }
            crate::qwen36::LayerDecodeState::Linear(cache) => {
                if let Some(array) = &cache.metal_conv_state {
                    array.eval()?;
                }
                if let Some(array) = &cache.metal_recurrent_state {
                    array.eval()?;
                }
            }
        }
    }
    Ok(())
}

fn eval_mtp_decode_state(state: &crate::qwen36::MtpDecodeState) -> Result<()> {
    eval_full_attention_cache(&state.cache)
}

fn eval_full_attention_cache(cache: &crate::qwen36::FullAttentionCache) -> Result<()> {
    if let Some(array) = &cache.k {
        array.eval()?;
    }
    if let Some(array) = &cache.v {
        array.eval()?;
    }
    Ok(())
}

pub(super) fn maybe_store_post_generation_cache(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    prompt_ids: &[u32],
    completion_ids: &[u32],
    mut state: crate::qwen36::DecodeState,
    start_hidden: mlx_rs::Array,
    mut mtp_history_state: crate::qwen36::MtpDecodeState,
) -> Result<()> {
    if !post_generation_cache_enabled() || completion_ids.is_empty() {
        return Ok(());
    }
    let combined_len = prompt_ids.len() + completion_ids.len();
    if combined_len > prefix_cache_max_tokens() {
        return Ok(());
    }
    let (logits, hidden) =
        session
            .qwen
            .decode_tokens_logits_with_hidden(completion_ids, &session.plan, &mut state)?;
    state.clear_transient_block_states();
    let last = completion_ids.len() as i32 - 1;
    let logits = logits.index((.., last..last + 1, ..));
    let last_hidden = hidden.index((.., last..last + 1, ..));
    if request.profile_timings {
        logits.eval()?;
        last_hidden.eval()?;
    }
    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
    if persistent_mtp && request.mtp && request.depth > 0 && session.qwen.mtp.is_some() {
        append_mtp_history_from_commit(
            session,
            &mut mtp_history_state,
            &start_hidden,
            completion_ids,
            &hidden,
        )?;
    }
    let mut cached_ids = Vec::with_capacity(combined_len);
    cached_ids.extend_from_slice(prompt_ids);
    cached_ids.extend_from_slice(completion_ids);
    maybe_store_prompt_cache(
        session,
        &cached_ids,
        &state,
        &logits,
        &last_hidden,
        &mtp_history_state,
    );
    Ok(())
}

pub(super) fn maybe_store_chat_post_generation_cache(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    assistant_text: &str,
) -> Result<()> {
    if !env_flag("MTPLX_CHAT_POST_GENERATION_CACHE", false)
        || !env_flag("MTPLX_PREFIX_CACHE", false)
        || assistant_text.is_empty()
    {
        return Ok(());
    }
    let mut messages = if request.messages.is_empty() {
        vec![PromptMessage {
            role: "user".to_string(),
            content: request.prompt.clone(),
        }]
    } else {
        request.messages.clone()
    };
    messages.push(PromptMessage {
        role: "assistant".to_string(),
        content: assistant_text.to_string(),
    });
    let prefix_ids = session
        .model
        .encode_chat_prefix(request.system.as_deref(), &messages)?;
    if prefix_ids.is_empty() || prefix_ids.len() > prefix_cache_max_tokens() {
        return Ok(());
    }
    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
    let needs_mtp_history =
        persistent_mtp && request.mtp && request.depth > 0 && session.qwen.mtp.is_some();
    let (state, logits, hidden, mtp_history_state) =
        prepare_prompt_state_chunked(session, request, &prefix_ids, needs_mtp_history)?;
    maybe_store_prompt_cache(
        session,
        &prefix_ids,
        &state,
        &logits,
        &hidden,
        &mtp_history_state,
    );
    Ok(())
}
