use super::super::{DEFAULT_MAX_KV_CONTEXT_TOKENS, env_flag, ferrite_env_var};

pub(super) fn prefix_cache_max_tokens() -> usize {
    ferrite_env_var("MTPLX_PREFIX_CACHE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(DEFAULT_MAX_KV_CONTEXT_TOKENS)
}

pub(super) fn prefill_chunk_tokens() -> usize {
    ferrite_env_var("MTPLX_PREFILL_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128)
}

pub(super) fn prefill_eval_interval_chunks() -> usize {
    ferrite_env_var("MTPLX_PREFILL_EVAL_INTERVAL_CHUNKS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(1)
}

pub(super) fn should_eval_prefill_chunk(chunk_index: usize, interval: usize) -> bool {
    (chunk_index + 1).is_multiple_of(interval)
}

pub(super) fn prefix_cache_max_entries() -> usize {
    ferrite_env_var("MTPLX_PREFIX_CACHE_ENTRIES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1)
}

pub(super) fn prefix_cache_max_bytes() -> usize {
    ferrite_env_var("MTPLX_PREFIX_CACHE_MAX_BYTES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1024 * 1024 * 1024)
}

pub(in crate::inference) fn post_generation_cache_enabled() -> bool {
    env_flag("MTPLX_POST_GENERATION_CACHE", false) && env_flag("MTPLX_PREFIX_CACHE", false)
}
