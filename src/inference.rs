#[cfg(feature = "native-mlx")]
use std::cell::RefCell;
use std::time::Instant;

use anyhow::Result;
use serde::{Deserialize, Serialize};

#[cfg(feature = "native-mlx")]
const DEFAULT_MAX_KV_CONTEXT_TOKENS: usize = 16_384;

#[cfg(feature = "native-mlx")]
mod memory;
#[cfg(feature = "native-mlx")]
use memory::{
    RequestMemoryGuard, check_kill_switch, clear_mlx_cache, log_memory_sample,
    memory_trace_enabled, request_memory_details,
};
#[cfg(feature = "native-mlx")]
mod prefill;
#[cfg(feature = "native-mlx")]
use prefill::{
    PromptStateCache, maybe_store_chat_post_generation_cache, maybe_store_post_generation_cache,
    post_generation_cache_enabled, prepare_prompt_state,
};
#[cfg(feature = "native-mlx")]
mod mtp;
#[cfg(feature = "native-mlx")]
use mtp::{StreamState, emit_stream_delta, generate_mtp};
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InferenceRequest {
    pub model: String,
    pub prompt: String,
    pub system: Option<String>,
    #[serde(default)]
    pub messages: Vec<PromptMessage>,
    #[serde(default)]
    pub stop: Vec<String>,
    pub max_tokens: Option<u32>,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub depth: u32,
    pub mtp: bool,
    #[serde(default)]
    pub requested_context_tokens: Option<u32>,
    #[serde(default)]
    pub profile_timings: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptMessage {
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct InferenceResponse {
    pub text: String,
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub finish_reason: FinishReason,
    pub backend: String,
    pub model: Option<crate::model::ModelSummary>,
    pub timings: Option<InferenceTimings>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtp: Option<MtpStats>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FinishReason {
    Stop,
    Length,
}

#[derive(Clone, Debug, Serialize)]
pub struct InferenceTimings {
    pub cache_hit: bool,
    pub prefix_cache_hit: bool,
    pub prefix_cache_tokens: usize,
    pub prefix_cache_entries: usize,
    pub load_s: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub total_s: f64,
    pub output_tokens_per_s: f64,
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct MtpStats {
    pub depth: u32,
    pub final_depth: u32,
    pub cycles: u32,
    pub all_accepted_cycles: u32,
    pub rejected_cycles: u32,
    pub generated_tokens_per_cycle: f64,
    pub draft_tokens: u32,
    pub accepted_draft_tokens: u32,
    pub rejected_draft_tokens: u32,
    pub acceptance_rate: f64,
    pub drafted_by_depth: Vec<u32>,
    pub accepted_by_depth: Vec<u32>,
    pub acceptance_by_depth: Vec<Option<f64>>,
    pub fallback_to_target: bool,
    pub correction_cache_hits: u32,
    pub correction_cache_misses: u32,
    pub lazy_verify_bonus_logits: u32,
    pub draft_s: f64,
    pub verify_s: f64,
    pub target_distribution_s: f64,
    pub commit_s: f64,
    pub bonus_s: f64,
    pub eager_verify_timing: bool,
    pub profile_eval_timing: bool,
    pub verify_profiled_blocks: u32,
    pub verify_profiled_tokens: u32,
    pub verify_embedding_s: f64,
    pub verify_full_attention_s: f64,
    pub verify_linear_attention_s: f64,
    pub verify_mlp_s: f64,
    pub verify_layer_glue_s: f64,
    pub verify_final_norm_s: f64,
    pub verify_lm_head_s: f64,
}

pub trait InferenceBackend {
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse>;
}

#[derive(Debug)]
pub struct NativeMlxBackend {
    #[cfg(feature = "native-mlx")]
    cache: RefCell<Option<NativeMlxSession>>,
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
struct NativeMlxSession {
    model_ref: String,
    model: crate::model::LoadedModel,
    weights: crate::mlx_backend::MlxWeightStore,
    qwen: crate::qwen36::Qwen36Weights,
    plan: crate::qwen36::Qwen36Plan,
    prompt_caches: Vec<PromptStateCache>,
}

impl NativeMlxBackend {
    pub fn new() -> Self {
        Self {
            #[cfg(feature = "native-mlx")]
            cache: RefCell::new(None),
        }
    }

    #[cfg(feature = "native-mlx")]
    pub fn preload(&self, model_ref: &str) -> Result<f64> {
        let _memory_guard = RequestMemoryGuard::new("preload");
        check_kill_switch("preload_start")?;
        let started = Instant::now();
        let mut cache = self.cache.borrow_mut();
        let needs_load = cache
            .as_ref()
            .map(|session| session.model_ref != model_ref)
            .unwrap_or(true);
        if needs_load {
            *cache = Some(NativeMlxSession::load(model_ref)?);
        }
        if memory_trace_enabled() {
            log_memory_sample("preload", "after_load", &format!("model={model_ref}"));
        }
        check_kill_switch("preload_after_load")?;
        Ok(started.elapsed().as_secs_f64())
    }

    #[cfg(not(feature = "native-mlx"))]
    pub fn preload(&self, model_ref: &str) -> Result<f64> {
        let _ = model_ref;
        Ok(0.0)
    }

    pub fn status() -> &'static str {
        "native Rust MLX forward path implemented with decode cache, custom Metal GDN kernels, and MTP speculative decoding"
    }
}

impl Default for NativeMlxBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "native-mlx")]
impl NativeMlxSession {
    fn load(model_ref: &str) -> Result<Self> {
        let model = crate::model::LoadedModel::load(model_ref)?;
        let weights =
            crate::mlx_backend::MlxWeightStore::load_model_dir(&model.path, &model.tensors)?;
        let qwen = crate::qwen36::Qwen36Weights::from_loaded(&model, &weights)?;
        let plan = crate::qwen36::Qwen36Plan::from_model(&model)?;
        Ok(Self {
            model_ref: model_ref.to_string(),
            model,
            weights,
            qwen,
            plan,
            prompt_caches: Vec::new(),
        })
    }
}

#[cfg(feature = "native-mlx")]
fn env_enabled(name: &str) -> bool {
    env_flag(name, false)
}

#[cfg(feature = "native-mlx")]
fn env_flag(name: &str, default: bool) -> bool {
    ferrite_env_var(name)
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            match value.as_str() {
                "1" | "true" | "yes" | "on" | "all" => true,
                "0" | "false" | "no" | "off" | "none" => false,
                _ => default,
            }
        })
        .unwrap_or(default)
}

#[cfg(feature = "native-mlx")]
fn ferrite_env_var(name: &str) -> std::result::Result<String, std::env::VarError> {
    if let Some(rest) = name.strip_prefix("MTPLX_") {
        if let Ok(value) = std::env::var(format!("FERRITE_{rest}")) {
            return Ok(value);
        }
    }
    std::env::var(name)
}

#[cfg(feature = "native-mlx")]
fn max_kv_context_tokens() -> usize {
    ferrite_env_var("MTPLX_MAX_KV_CONTEXT_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(DEFAULT_MAX_KV_CONTEXT_TOKENS)
}

#[cfg(feature = "native-mlx")]
fn bounded_max_tokens(
    prompt_tokens: usize,
    requested_max_tokens: u32,
    max_context_tokens: usize,
) -> Result<u32> {
    if requested_max_tokens == 0 {
        return Ok(0);
    }
    if prompt_tokens >= max_context_tokens {
        anyhow::bail!(
            "request context too large: prompt has {prompt_tokens} tokens, max KV context is {max_context_tokens}; reduce input or set FERRITE_MAX_KV_CONTEXT_TOKENS"
        );
    }
    let remaining = max_context_tokens - prompt_tokens;
    Ok(requested_max_tokens.min(remaining.min(u32::MAX as usize) as u32))
}

impl NativeMlxBackend {
    pub fn infer_stream<F>(
        &self,
        request: &InferenceRequest,
        mut on_delta: F,
    ) -> Result<InferenceResponse>
    where
        F: FnMut(&str) -> Result<()>,
    {
        #[cfg(feature = "native-mlx")]
        {
            let mut stream = StreamState {
                emitted_text: String::new(),
                on_delta: &mut on_delta,
            };
            self.infer_internal(request, Some(&mut stream))
        }
        #[cfg(not(feature = "native-mlx"))]
        {
            let response = self.infer(request)?;
            on_delta(&response.text)?;
            Ok(response)
        }
    }

    fn infer_internal(
        &self,
        request: &InferenceRequest,
        #[cfg(feature = "native-mlx")] mut stream: Option<&mut StreamState<'_>>,
    ) -> Result<InferenceResponse> {
        let total_started = Instant::now();
        #[cfg(feature = "native-mlx")]
        {
            let _memory_guard = RequestMemoryGuard::new("infer");
            check_kill_switch("infer_start")?;
            let load_started = Instant::now();
            let mut cache = self.cache.borrow_mut();
            let needs_load = cache
                .as_ref()
                .map(|session| session.model_ref != request.model)
                .unwrap_or(true);
            if needs_load {
                *cache = Some(NativeMlxSession::load(&request.model)?);
            }
            let session = cache
                .as_mut()
                .ok_or_else(|| anyhow::anyhow!("native MLX model cache was not initialized"))?;
            let load_s = load_started.elapsed().as_secs_f64();
            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "after_load",
                    &request_memory_details(request, None, None),
                );
            }
            check_kill_switch("infer_after_load")?;
            let prompt_ids = session.model.encode_prompt(request)?;
            let max_tokens = bounded_max_tokens(
                prompt_ids.len(),
                request.max_tokens.unwrap_or(1),
                max_kv_context_tokens(),
            )?;
            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "after_tokenize",
                    &request_memory_details(request, Some(prompt_ids.len()), Some(max_tokens)),
                );
            }
            check_kill_switch("infer_after_tokenize")?;
            let eos_token_ids = session.model.eos_token_ids();
            let sampling = crate::sampling::SamplingConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            };
            let mut completion_ids = Vec::new();
            let mut finish_reason = FinishReason::Length;
            let mut prefill_s = 0.0;
            let mut prefix_cache_hit = false;
            let mut prefix_cache_tokens = 0_usize;
            let mut post_generation_cache_base = None;
            let decode_started = Instant::now();
            if max_tokens > 0 {
                check_kill_switch("infer_before_prefill")?;
                let prepared = prepare_prompt_state(session, request, &prompt_ids)?;
                let mut state = prepared.state;
                let logits = prepared.logits;
                let hidden = prepared.hidden;
                let mtp_history_state = prepared.mtp_history_state;
                prefill_s = prepared.prefill_s;
                prefix_cache_hit = prepared.prefix_cache_hit;
                prefix_cache_tokens = prepared.prefix_cache_tokens;
                if memory_trace_enabled() {
                    log_memory_sample(
                        "infer",
                        "after_prefill",
                        &request_memory_details(request, Some(prompt_ids.len()), Some(max_tokens)),
                    );
                }
                check_kill_switch("infer_after_prefill")?;
                if request.mtp && request.depth > 0 && session.qwen.mtp.is_some() {
                    let post_generation_cache_base = post_generation_cache_enabled()
                        .then(|| (state.clone(), hidden.clone(), mtp_history_state.clone()));
                    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
                    let generated = generate_mtp(
                        session,
                        request,
                        &eos_token_ids,
                        sampling,
                        max_tokens,
                        state,
                        logits,
                        hidden,
                        mtp_history_state,
                        persistent_mtp,
                        stream.as_deref_mut(),
                    )?;
                    let decode_s = (decode_started.elapsed().as_secs_f64() - prefill_s).max(0.0);
                    let stopped_on_text = generated.stopped_text.is_some();
                    let text = match generated.stopped_text {
                        Some(text) => text,
                        None => session
                            .model
                            .tokenizer
                            .decode(&generated.completion_ids, true)
                            .map_err(|err| anyhow::anyhow!("decode completion: {err}"))?,
                    };
                    if !stopped_on_text
                        && let Some((
                            cache_base_state,
                            cache_base_hidden,
                            cache_base_mtp_history_state,
                        )) = post_generation_cache_base
                    {
                        maybe_store_post_generation_cache(
                            session,
                            request,
                            &prompt_ids,
                            &generated.completion_ids,
                            cache_base_state,
                            cache_base_hidden,
                            cache_base_mtp_history_state,
                        )?;
                    }
                    maybe_store_chat_post_generation_cache(session, request, &text)?;
                    let _ = (session.weights.len(), session.weights.source_files.len());
                    let total_s = total_started.elapsed().as_secs_f64();
                    if memory_trace_enabled() {
                        log_memory_sample(
                            "infer",
                            "before_return_mtp",
                            &request_memory_details(
                                request,
                                Some(prompt_ids.len()),
                                Some(max_tokens),
                            ),
                        );
                    }
                    return Ok(InferenceResponse {
                        text,
                        prompt_tokens: prompt_ids.len() as u32,
                        completion_tokens: generated.completion_ids.len() as u32,
                        finish_reason: generated.finish_reason,
                        backend: "native-mlx-rust-mtp-speculative".to_string(),
                        model: Some(session.model.summary()),
                        timings: Some(InferenceTimings {
                            cache_hit: !needs_load,
                            prefix_cache_hit,
                            prefix_cache_tokens,
                            prefix_cache_entries: session.prompt_caches.len(),
                            load_s,
                            prefill_s,
                            decode_s,
                            total_s,
                            output_tokens_per_s: generated.completion_ids.len() as f64
                                / decode_s.max(f64::EPSILON),
                        }),
                        mtp: generated.mtp,
                    });
                }
                post_generation_cache_base = post_generation_cache_enabled()
                    .then(|| (state.clone(), hidden.clone(), mtp_history_state.clone()));
                let mut next = crate::sampling::next_from_logits(&logits, sampling)?;
                if eos_token_ids.contains(&next) {
                    let total_s = total_started.elapsed().as_secs_f64();
                    if memory_trace_enabled() {
                        log_memory_sample(
                            "infer",
                            "before_return_eos",
                            &request_memory_details(
                                request,
                                Some(prompt_ids.len()),
                                Some(max_tokens),
                            ),
                        );
                    }
                    return Ok(InferenceResponse {
                        text: String::new(),
                        prompt_tokens: prompt_ids.len() as u32,
                        completion_tokens: 0,
                        finish_reason: FinishReason::Stop,
                        backend: "native-mlx-rust-metal-gdn-cached".to_string(),
                        model: Some(session.model.summary()),
                        timings: Some(InferenceTimings {
                            cache_hit: !needs_load,
                            prefix_cache_hit,
                            prefix_cache_tokens,
                            prefix_cache_entries: session.prompt_caches.len(),
                            load_s,
                            prefill_s,
                            decode_s: 0.0,
                            total_s,
                            output_tokens_per_s: 0.0,
                        }),
                        mtp: None,
                    });
                }
                completion_ids.push(next);
                emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
                let mut stopped_text =
                    stop_text_if_needed(&session.model, &completion_ids, &request.stop)?;
                if stopped_text.is_some() {
                    finish_reason = FinishReason::Stop;
                }
                for _ in 1..max_tokens {
                    check_kill_switch("infer_decode_loop")?;
                    if stopped_text.is_some() {
                        break;
                    }
                    let logits =
                        session
                            .qwen
                            .decode_step_logits(next, &session.plan, &mut state)?;
                    if request.profile_timings {
                        logits.eval()?;
                    }
                    next = crate::sampling::next_from_logits(&logits, sampling)?;
                    if eos_token_ids.contains(&next) {
                        finish_reason = FinishReason::Stop;
                        break;
                    }
                    completion_ids.push(next);
                    emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
                    stopped_text =
                        stop_text_if_needed(&session.model, &completion_ids, &request.stop)?;
                    if stopped_text.is_some() {
                        finish_reason = FinishReason::Stop;
                        break;
                    }
                }
                if let Some(text) = stopped_text {
                    let decode_s = (decode_started.elapsed().as_secs_f64() - prefill_s).max(0.0);
                    maybe_store_chat_post_generation_cache(session, request, &text)?;
                    let _ = (session.weights.len(), session.weights.source_files.len());
                    let total_s = total_started.elapsed().as_secs_f64();
                    if memory_trace_enabled() {
                        log_memory_sample(
                            "infer",
                            "before_return_stop",
                            &request_memory_details(
                                request,
                                Some(prompt_ids.len()),
                                Some(max_tokens),
                            ),
                        );
                    }
                    return Ok(InferenceResponse {
                        text,
                        prompt_tokens: prompt_ids.len() as u32,
                        completion_tokens: completion_ids.len() as u32,
                        finish_reason,
                        backend: "native-mlx-rust-metal-gdn-cached".to_string(),
                        model: Some(session.model.summary()),
                        timings: Some(InferenceTimings {
                            cache_hit: !needs_load,
                            prefix_cache_hit,
                            prefix_cache_tokens,
                            prefix_cache_entries: session.prompt_caches.len(),
                            load_s,
                            prefill_s,
                            decode_s,
                            total_s,
                            output_tokens_per_s: completion_ids.len() as f64
                                / decode_s.max(f64::EPSILON),
                        }),
                        mtp: None,
                    });
                }
            }
            let decode_s = (decode_started.elapsed().as_secs_f64() - prefill_s).max(0.0);
            let text = session
                .model
                .tokenizer
                .decode(&completion_ids, true)
                .map_err(|err| anyhow::anyhow!("decode completion: {err}"))?;
            if let Some((cache_base_state, cache_base_hidden, cache_base_mtp_history_state)) =
                post_generation_cache_base
            {
                maybe_store_post_generation_cache(
                    session,
                    request,
                    &prompt_ids,
                    &completion_ids,
                    cache_base_state,
                    cache_base_hidden,
                    cache_base_mtp_history_state,
                )?;
            }
            maybe_store_chat_post_generation_cache(session, request, &text)?;
            let _ = (session.weights.len(), session.weights.source_files.len());
            let total_s = total_started.elapsed().as_secs_f64();
            if memory_trace_enabled() {
                log_memory_sample(
                    "infer",
                    "before_return",
                    &request_memory_details(request, Some(prompt_ids.len()), Some(max_tokens)),
                );
            }
            Ok(InferenceResponse {
                text,
                prompt_tokens: prompt_ids.len() as u32,
                completion_tokens: completion_ids.len() as u32,
                finish_reason,
                backend: "native-mlx-rust-metal-gdn-cached".to_string(),
                model: Some(session.model.summary()),
                timings: Some(InferenceTimings {
                    cache_hit: !needs_load,
                    prefix_cache_hit,
                    prefix_cache_tokens,
                    prefix_cache_entries: session.prompt_caches.len(),
                    load_s,
                    prefill_s,
                    decode_s,
                    total_s,
                    output_tokens_per_s: completion_ids.len() as f64 / decode_s.max(f64::EPSILON),
                }),
                mtp: None,
            })
        }
        #[cfg(not(feature = "native-mlx"))]
        {
            let model = crate::model::LoadedModel::load(&request.model)?;
            let prompt_ids = model.encode_prompt(request)?;
            anyhow::bail!(
                "{}; loaded {} prompt tokens for {} ({:?})",
                Self::status(),
                prompt_ids.len(),
                model.path.display(),
                model.config.architecture()
            )
        }
    }
}

impl InferenceBackend for NativeMlxBackend {
    fn infer(&self, request: &InferenceRequest) -> Result<InferenceResponse> {
        self.infer_internal(
            request,
            #[cfg(feature = "native-mlx")]
            None,
        )
    }
}

#[cfg(feature = "native-mlx")]
fn stop_text_if_needed(
    model: &crate::model::LoadedModel,
    completion_ids: &[u32],
    stop: &[String],
) -> Result<Option<String>> {
    if stop.is_empty() {
        return Ok(None);
    }
    let text = model
        .tokenizer
        .decode(completion_ids, true)
        .map_err(|err| anyhow::anyhow!("decode completion for stop check: {err}"))?;
    Ok(trim_at_stop(text, stop))
}

fn trim_at_stop(text: String, stop: &[String]) -> Option<String> {
    let index = stop
        .iter()
        .filter(|value| !value.is_empty())
        .filter_map(|value| text.find(value))
        .min()?;
    Some(text[..index].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trim_at_stop_uses_earliest_match() {
        let text = trim_at_stop(
            "alpha beta gamma".to_string(),
            &["gamma".to_string(), " beta".to_string()],
        )
        .unwrap();
        assert_eq!(text, "alpha");
    }

    #[test]
    fn trim_at_stop_ignores_empty_stops() {
        assert_eq!(trim_at_stop("alpha".to_string(), &["".to_string()]), None);
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn bounded_max_tokens_respects_context_limit() {
        assert_eq!(bounded_max_tokens(60, 16, 64).unwrap(), 4);
        assert_eq!(bounded_max_tokens(64, 0, 64).unwrap(), 0);
        assert!(bounded_max_tokens(64, 1, 64).is_err());
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn memory_log_appends_to_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ferrite-memory.log");

        memory::append_memory_log_line_to(&path, "ferrite_memory phase=test one=1").unwrap();
        memory::append_memory_log_line_to(&path, "ferrite_memory phase=test two=2").unwrap();

        let text = std::fs::read_to_string(path).unwrap();
        assert!(text.contains("phase=test one=1"));
        assert!(text.contains("phase=test two=2"));
    }
}
