use std::sync::OnceLock;

use super::super::{DEFAULT_MAX_KV_CONTEXT_TOKENS, env_flag, ferrite_env_var};

const DEFAULT_DENSE_PREFILL_CHUNK_TOKENS: usize = 2048;
const DEFAULT_REPAGE_PREFILL_CHUNK_TOKENS: usize = 2048;
const DEFAULT_DENSE_PREFILL_MAX_CONTEXT_TOKENS: usize = 65_536;

pub(super) fn prefix_cache_max_tokens() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFIX_CACHE_MAX_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_MAX_KV_CONTEXT_TOKENS)
    })
}

pub(super) fn prefill_chunk_tokens(prompt_tokens: usize) -> usize {
    if let Some(value) = prefill_chunk_override() {
        let trimmed = value.trim();
        if trimmed.eq_ignore_ascii_case("auto") {
            return auto_prefill_chunk_tokens(prompt_tokens);
        }
        if let Ok(parsed) = trimmed.parse::<usize>()
            && parsed > 0
        {
            return parsed;
        }
    }
    auto_prefill_chunk_tokens(prompt_tokens)
}

fn prefill_chunk_override() -> Option<String> {
    ferrite_env_var("MTPLX_PREFILL_CHUNK_TOKENS")
        .ok()
        .or_else(|| ferrite_env_var("MTPLX_PREFILL_CHUNK_SIZE").ok())
}

fn auto_prefill_chunk_tokens(prompt_tokens: usize) -> usize {
    if prompt_tokens <= dense_prefill_max_context_tokens() {
        dense_prefill_chunk_tokens()
    } else {
        repage_prefill_chunk_tokens()
    }
}

fn dense_prefill_chunk_tokens() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFILL_CHUNK_SIZE_DENSE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_DENSE_PREFILL_CHUNK_TOKENS)
    })
}

fn repage_prefill_chunk_tokens() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFILL_CHUNK_SIZE_REPAGE")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_REPAGE_PREFILL_CHUNK_TOKENS)
    })
}

fn dense_prefill_max_context_tokens() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_SUSTAINED_DENSE_DECODE_MAX_CONTEXT")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(DEFAULT_DENSE_PREFILL_MAX_CONTEXT_TOKENS)
    })
}

pub(super) fn prefill_eval_interval_chunks() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFILL_EVAL_INTERVAL_CHUNKS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(1)
    })
}

pub(super) fn should_eval_prefill_chunk(chunk_index: usize, interval: usize) -> bool {
    (chunk_index + 1).is_multiple_of(interval)
}

pub(super) fn prefix_cache_max_entries() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFIX_CACHE_ENTRIES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1)
    })
}

pub(super) fn prefix_cache_max_bytes() -> usize {
    static VALUE: OnceLock<usize> = OnceLock::new();
    *VALUE.get_or_init(|| {
        ferrite_env_var("MTPLX_PREFIX_CACHE_MAX_BYTES")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1024 * 1024 * 1024)
    })
}

pub(in crate::inference) fn post_generation_cache_enabled() -> bool {
    static VALUE: OnceLock<bool> = OnceLock::new();
    *VALUE.get_or_init(|| {
        env_flag("MTPLX_POST_GENERATION_CACHE", false) && env_flag("MTPLX_PREFIX_CACHE", false)
    })
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_DENSE_PREFILL_CHUNK_TOKENS, DEFAULT_DENSE_PREFILL_MAX_CONTEXT_TOKENS,
        DEFAULT_REPAGE_PREFILL_CHUNK_TOKENS, auto_prefill_chunk_tokens,
    };

    #[test]
    fn auto_prefill_uses_dense_chunks_up_to_dense_context_limit() {
        assert_eq!(
            auto_prefill_chunk_tokens(DEFAULT_DENSE_PREFILL_MAX_CONTEXT_TOKENS),
            DEFAULT_DENSE_PREFILL_CHUNK_TOKENS
        );
    }

    #[test]
    fn auto_prefill_uses_repage_chunks_above_dense_context_limit() {
        assert_eq!(
            auto_prefill_chunk_tokens(DEFAULT_DENSE_PREFILL_MAX_CONTEXT_TOKENS + 1),
            DEFAULT_REPAGE_PREFILL_CHUNK_TOKENS
        );
    }
}
