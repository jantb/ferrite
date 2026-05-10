#[cfg(feature = "native-mlx")]
use std::cell::RefCell;
use std::time::Instant;

use anyhow::Result;
#[cfg(feature = "native-mlx")]
use mlx_rs::ops::indexing::IndexOp;
use serde::{Deserialize, Serialize};

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

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug)]
struct PromptStateCache {
    prompt_ids: Vec<u32>,
    state: crate::qwen36::DecodeState,
    logits: mlx_rs::Array,
    hidden: mlx_rs::Array,
    mtp_history_state: crate::qwen36::MtpDecodeState,
}

#[cfg(feature = "native-mlx")]
struct PreparedPrompt {
    state: crate::qwen36::DecodeState,
    logits: mlx_rs::Array,
    hidden: mlx_rs::Array,
    mtp_history_state: crate::qwen36::MtpDecodeState,
    prefill_s: f64,
    prefix_cache_hit: bool,
    prefix_cache_tokens: usize,
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
        let started = Instant::now();
        let mut cache = self.cache.borrow_mut();
        let needs_load = cache
            .as_ref()
            .map(|session| session.model_ref != model_ref)
            .unwrap_or(true);
        if needs_load {
            *cache = Some(NativeMlxSession::load(model_ref)?);
        }
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
struct GenerationOutput {
    completion_ids: Vec<u32>,
    finish_reason: FinishReason,
    stopped_text: Option<String>,
    mtp: Option<MtpStats>,
}

#[cfg(feature = "native-mlx")]
struct StreamState<'a> {
    emitted_text: String,
    on_delta: &'a mut dyn FnMut(&str) -> Result<()>,
}

#[cfg(feature = "native-mlx")]
impl<'a> StreamState<'a> {
    fn emit(&mut self, model: &crate::model::LoadedModel, completion_ids: &[u32]) -> Result<()> {
        let text = model
            .tokenizer
            .decode(completion_ids, true)
            .map_err(|err| anyhow::anyhow!("decode stream delta: {err}"))?;
        let delta = text
            .strip_prefix(&self.emitted_text)
            .unwrap_or(text.as_str());
        if !delta.is_empty() {
            (self.on_delta)(delta)?;
        }
        self.emitted_text = text;
        Ok(())
    }
}

#[cfg(feature = "native-mlx")]
fn emit_stream_delta(
    stream: &mut Option<&mut StreamState<'_>>,
    model: &crate::model::LoadedModel,
    completion_ids: &[u32],
) -> Result<()> {
    if let Some(stream) = stream.as_deref_mut() {
        stream.emit(model, completion_ids)?;
    }
    Ok(())
}

#[cfg(feature = "native-mlx")]
enum DraftCandidate {
    Greedy(u32),
    Sampled {
        token: u32,
        distribution: crate::sampling::TokenDistribution,
    },
}

#[cfg(feature = "native-mlx")]
impl DraftCandidate {
    fn token(&self) -> u32 {
        match self {
            Self::Greedy(token) => *token,
            Self::Sampled { token, .. } => *token,
        }
    }

    fn probability(&self, token: u32) -> f64 {
        match self {
            Self::Greedy(greedy_token) => {
                if *greedy_token == token {
                    1.0
                } else {
                    0.0
                }
            }
            Self::Sampled { distribution, .. } => distribution.probability(token),
        }
    }

    fn distribution(&self) -> crate::sampling::TokenDistribution {
        match self {
            Self::Greedy(token) => crate::sampling::TokenDistribution::deterministic(*token),
            Self::Sampled { distribution, .. } => distribution.clone(),
        }
    }
}

#[cfg(feature = "native-mlx")]
fn generate_mtp(
    session: &NativeMlxSession,
    request: &InferenceRequest,
    eos_token_ids: &[u32],
    sampling: crate::sampling::SamplingConfig,
    max_tokens: u32,
    mut state: crate::qwen36::DecodeState,
    mut logits: mlx_rs::Array,
    mut hidden: mlx_rs::Array,
    mut mtp_history_state: crate::qwen36::MtpDecodeState,
    persistent_mtp: bool,
    mut stream: Option<&mut StreamState<'_>>,
) -> Result<GenerationOutput> {
    let mut completion_ids = Vec::with_capacity(max_tokens as usize);
    let mut finish_reason = FinishReason::Length;
    let mut pending_primary = None;
    let mut stats = MtpStats {
        depth: request.depth,
        final_depth: request.depth,
        drafted_by_depth: vec![0; request.depth as usize],
        accepted_by_depth: vec![0; request.depth as usize],
        acceptance_by_depth: vec![None; request.depth as usize],
        ..MtpStats::default()
    };
    let profile_eval_timing = request.profile_timings || env_enabled("MTPLX_PROFILE_EVAL_TIMING");
    stats.profile_eval_timing = profile_eval_timing;
    let mut active_depth = request.depth.max(1);
    let draft_sampling = if sampling.temperature > 0.0 {
        crate::sampling::SamplingConfig {
            temperature: 0.7,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
        }
    } else {
        sampling
    };

    while completion_ids.len() < max_tokens as usize {
        let primary = match pending_primary.take() {
            Some(token) => token,
            None => {
                let token =
                    crate::sampling::distribution_from_logits(&logits, sampling)?.sample()?;
                if eos_token_ids.contains(&token) {
                    finish_reason = FinishReason::Stop;
                    break;
                }
                completion_ids.push(token);
                emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
                if let Some(text) =
                    stop_text_if_needed(&session.model, &completion_ids, &request.stop)?
                {
                    finish_reason = FinishReason::Stop;
                    return Ok(finalize_mtp_generation(
                        completion_ids,
                        finish_reason,
                        Some(text),
                        stats,
                    ));
                }
                token
            }
        };
        if completion_ids.len() >= max_tokens as usize {
            break;
        }

        let mut mtp_state = if persistent_mtp {
            mtp_history_state.clone()
        } else {
            session.qwen.new_mtp_decode_state()?
        };
        stats.cycles += 1;
        let mut draft_hidden = hidden.clone();
        let mut draft_input = primary;
        let mut drafts = Vec::new();
        let draft_start = Instant::now();
        for depth_index in 0..active_depth {
            if completion_ids.len() + drafts.len() >= max_tokens as usize {
                break;
            }
            let (draft_logits, next_draft_hidden) = session.qwen.mtp_draft_step(
                draft_input,
                &draft_hidden,
                &session.plan,
                &mut mtp_state,
            )?;
            if profile_eval_timing {
                draft_logits.eval()?;
                next_draft_hidden.eval()?;
            }
            let candidate = if draft_sampling.is_greedy() {
                DraftCandidate::Greedy(crate::sampling::greedy_token_from_logits(&draft_logits)?)
            } else {
                let distribution =
                    crate::sampling::distribution_from_logits(&draft_logits, draft_sampling)?;
                let token = distribution.sample()?;
                DraftCandidate::Sampled {
                    token,
                    distribution,
                }
            };
            let draft = candidate.token();
            drafts.push(candidate);
            stats.draft_tokens += 1;
            if let Some(count) = stats.drafted_by_depth.get_mut(depth_index as usize) {
                *count += 1;
            }
            draft_input = draft;
            draft_hidden = next_draft_hidden;
        }
        stats.draft_s += draft_start.elapsed().as_secs_f64();

        let mut verify_tokens = Vec::with_capacity(drafts.len() + 1);
        verify_tokens.push(primary);
        verify_tokens.extend(drafts.iter().map(DraftCandidate::token));
        let verify_base_position = state.position;
        let mut verify_state = state.clone();
        let verify_start = Instant::now();
        let (verified_block_logits, verified_block_hidden) = if profile_eval_timing {
            let (logits, hidden, profile) =
                session.qwen.decode_tokens_logits_with_hidden_profiled(
                    &verify_tokens,
                    &session.plan,
                    &mut verify_state,
                )?;
            stats.verify_profiled_blocks += profile.blocks;
            stats.verify_profiled_tokens += profile.tokens;
            stats.verify_embedding_s += profile.embedding_s;
            stats.verify_full_attention_s += profile.full_attention_s;
            stats.verify_linear_attention_s += profile.linear_attention_s;
            stats.verify_mlp_s += profile.mlp_s;
            stats.verify_layer_glue_s += profile.layer_glue_s;
            stats.verify_final_norm_s += profile.final_norm_s;
            stats.verify_lm_head_s += profile.lm_head_s;
            (logits, hidden)
        } else if env_enabled("MTPLX_LAZY_VERIFY_LOGITS") {
            let hidden = session.qwen.decode_tokens_hidden(
                &verify_tokens,
                &session.plan,
                &mut verify_state,
            )?;
            let draft_rows = drafts.len() as i32;
            let logits = session
                .qwen
                .logits_from_hidden(&hidden.index((.., 0..draft_rows, ..)))?;
            (logits, hidden)
        } else {
            session.qwen.decode_tokens_logits_with_hidden(
                &verify_tokens,
                &session.plan,
                &mut verify_state,
            )?
        };
        if profile_eval_timing || env_enabled("MTPLX_EAGER_VERIFY_TIMING") {
            verified_block_logits.eval()?;
            stats.eager_verify_timing = true;
        }
        if profile_eval_timing {
            verified_block_hidden.eval()?;
        }
        stats.verify_s += verify_start.elapsed().as_secs_f64();
        let verified_logit_rows = verified_block_logits.shape()[1];

        let mut accepted = Vec::new();
        let mut correction = None;
        let mut all_accepted = true;
        let mut stop_after_commit = false;
        let mut last_committed_index = 0_i32;
        if sampling.is_greedy() && draft_sampling.is_greedy() {
            let target_tokens = crate::sampling::greedy_tokens_from_logits(&verified_block_logits)?;
            for (index, candidate) in drafts.into_iter().enumerate() {
                let draft = candidate.token();
                let target = target_tokens.get(index).copied().ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing greedy target token at index {index}; got {} tokens",
                        target_tokens.len()
                    )
                })?;
                if draft == target {
                    stats.accepted_draft_tokens += 1;
                    if let Some(count) = stats.accepted_by_depth.get_mut(index) {
                        *count += 1;
                    }
                    if eos_token_ids.contains(&draft) {
                        finish_reason = FinishReason::Stop;
                        stop_after_commit = true;
                        break;
                    }
                    accepted.push(draft);
                    last_committed_index = index as i32 + 1;
                } else {
                    stats.rejected_draft_tokens += 1;
                    all_accepted = false;
                    correction = Some(target);
                    break;
                }
            }
        } else {
            let distribution_start = Instant::now();
            let target_distributions =
                crate::sampling::distributions_from_logits(&verified_block_logits, sampling)?;
            stats.target_distribution_s += distribution_start.elapsed().as_secs_f64();
            for (index, candidate) in drafts.into_iter().enumerate() {
                let draft = candidate.token();
                let target_dist = target_distributions.get(index).ok_or_else(|| {
                    anyhow::anyhow!(
                        "missing target distribution at index {index}; got {} distributions",
                        target_distributions.len()
                    )
                })?;
                let target_probability = target_dist.probability(draft);
                let draft_probability = candidate.probability(draft);
                let accept_probability = if draft_probability <= 0.0 {
                    0.0
                } else {
                    (target_probability / draft_probability).min(1.0)
                };
                if rand::random::<f64>() <= accept_probability {
                    stats.accepted_draft_tokens += 1;
                    if let Some(count) = stats.accepted_by_depth.get_mut(index) {
                        *count += 1;
                    }
                    if eos_token_ids.contains(&draft) {
                        finish_reason = FinishReason::Stop;
                        stop_after_commit = true;
                        break;
                    }
                    accepted.push(draft);
                    last_committed_index = index as i32 + 1;
                } else {
                    stats.rejected_draft_tokens += 1;
                    all_accepted = false;
                    let residual = crate::sampling::TokenDistribution::residual_from(
                        target_dist,
                        &candidate.distribution(),
                    );
                    correction = Some(residual.sample()?);
                    break;
                }
            }
        }

        if all_accepted {
            stats.all_accepted_cycles += 1;
            let commit_start = Instant::now();
            for token in accepted {
                if completion_ids.len() >= max_tokens as usize {
                    break;
                }
                completion_ids.push(token);
                emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
                if let Some(text) =
                    stop_text_if_needed(&session.model, &completion_ids, &request.stop)?
                {
                    finish_reason = FinishReason::Stop;
                    return Ok(finalize_mtp_generation(
                        completion_ids,
                        finish_reason,
                        Some(text),
                        stats,
                    ));
                }
            }
            state = verify_state;
            state.clear_transient_block_states();
            hidden = verified_block_hidden.index((
                ..,
                last_committed_index..last_committed_index + 1,
                ..,
            ));
            logits = if last_committed_index < verified_logit_rows {
                verified_block_logits.index((
                    ..,
                    last_committed_index..last_committed_index + 1,
                    ..,
                ))
            } else {
                stats.lazy_verify_bonus_logits += 1;
                session.qwen.logits_from_hidden(&hidden)?
            };
            if profile_eval_timing {
                logits.eval()?;
                hidden.eval()?;
            }
            if persistent_mtp {
                mtp_history_state = mtp_state;
            }
            stats.commit_s += commit_start.elapsed().as_secs_f64();
            if stop_after_commit {
                break;
            }
            if should_fallback_to_target(&stats) {
                stats.fallback_to_target = true;
                stats.final_depth = active_depth;
                return generate_target_tail(
                    session,
                    request,
                    eos_token_ids,
                    sampling,
                    max_tokens,
                    completion_ids,
                    state,
                    logits,
                    stats,
                    stream,
                );
            }
            active_depth = adapt_mtp_depth(&stats, active_depth);
            stats.final_depth = active_depth;
            if completion_ids.len() < max_tokens as usize {
                let bonus_start = Instant::now();
                let bonus =
                    crate::sampling::distribution_from_logits(&logits, sampling)?.sample()?;
                stats.bonus_s += bonus_start.elapsed().as_secs_f64();
                if eos_token_ids.contains(&bonus) {
                    finish_reason = FinishReason::Stop;
                    break;
                }
                completion_ids.push(bonus);
                emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
                pending_primary = Some(bonus);
                if let Some(text) =
                    stop_text_if_needed(&session.model, &completion_ids, &request.stop)?
                {
                    finish_reason = FinishReason::Stop;
                    return Ok(finalize_mtp_generation(
                        completion_ids,
                        finish_reason,
                        Some(text),
                        stats,
                    ));
                }
            }
            continue;
        }

        let accepted_tokens = accepted;
        stats.rejected_cycles += 1;
        let commit_start = Instant::now();
        let mut state_tokens = Vec::with_capacity(accepted_tokens.len() + 2);
        state_tokens.push(primary);
        state_tokens.extend(accepted_tokens.iter().copied());
        let mut emitted_tokens = accepted_tokens;
        let mut correction_to_decode = None;
        if let Some(token) = correction {
            if eos_token_ids.contains(&token) {
                finish_reason = FinishReason::Stop;
                stop_after_commit = true;
            } else if completion_ids.len() + emitted_tokens.len() < max_tokens as usize {
                correction_to_decode = Some(token);
            }
        }

        let mut used_correction_cache = false;
        if env_flag("MTPLX_CORRECTION_CACHE", true) {
            let start_hidden_for_mtp = hidden.clone();
            let prefix_len = state_tokens.len() as i32;
            let mut cached_state = verify_state;
            match cached_state.truncate_after_decode_block(verify_base_position, prefix_len) {
                Ok(()) => {
                    let prefix_hidden = verified_block_hidden.index((.., 0..prefix_len, ..));
                    if let Some(token) = correction_to_decode {
                        let (correction_logits, correction_hidden) =
                            session.qwen.decode_step_logits_with_hidden(
                                token,
                                &session.plan,
                                &mut cached_state,
                            )?;
                        state_tokens.push(token);
                        emitted_tokens.push(token);
                        logits = correction_logits;
                        hidden = correction_hidden;
                    } else {
                        let last = prefix_len - 1;
                        logits = verified_block_logits.index((.., last..last + 1, ..));
                        hidden = verified_block_hidden.index((.., last..last + 1, ..));
                    }
                    if profile_eval_timing {
                        logits.eval()?;
                        hidden.eval()?;
                    }
                    state = cached_state;
                    state.clear_transient_block_states();
                    if persistent_mtp {
                        append_mtp_history_from_commit(
                            session,
                            &mut mtp_history_state,
                            &start_hidden_for_mtp,
                            &state_tokens,
                            &prefix_hidden,
                        )?;
                    }
                    stats.correction_cache_hits += 1;
                    used_correction_cache = true;
                }
                Err(_) => {
                    stats.correction_cache_misses += 1;
                }
            }
        }

        if !used_correction_cache {
            if let Some(token) = correction_to_decode {
                state_tokens.push(token);
                emitted_tokens.push(token);
            }
            let (committed_logits, committed_hidden) = session
                .qwen
                .decode_tokens_logits_with_hidden(&state_tokens, &session.plan, &mut state)?;
            if profile_eval_timing {
                committed_logits.eval()?;
                committed_hidden.eval()?;
            }
            if persistent_mtp {
                append_mtp_history_from_commit(
                    session,
                    &mut mtp_history_state,
                    &hidden,
                    &state_tokens,
                    &committed_hidden,
                )?;
            }
            let last = state_tokens.len() as i32 - 1;
            logits = committed_logits.index((.., last..last + 1, ..));
            hidden = committed_hidden.index((.., last..last + 1, ..));
        }
        stats.commit_s += commit_start.elapsed().as_secs_f64();
        for token in emitted_tokens {
            completion_ids.push(token);
            emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
            if let Some(text) = stop_text_if_needed(&session.model, &completion_ids, &request.stop)?
            {
                finish_reason = FinishReason::Stop;
                return Ok(finalize_mtp_generation(
                    completion_ids,
                    finish_reason,
                    Some(text),
                    stats,
                ));
            }
            if completion_ids.len() >= max_tokens as usize {
                break;
            }
        }
        if stop_after_commit {
            break;
        }
        if should_fallback_to_target(&stats) {
            stats.fallback_to_target = true;
            stats.final_depth = active_depth;
            return generate_target_tail(
                session,
                request,
                eos_token_ids,
                sampling,
                max_tokens,
                completion_ids,
                state,
                logits,
                stats,
                stream,
            );
        }
        active_depth = adapt_mtp_depth(&stats, active_depth);
        stats.final_depth = active_depth;
    }

    Ok(finalize_mtp_generation(
        completion_ids,
        finish_reason,
        None,
        stats,
    ))
}

#[cfg(feature = "native-mlx")]
fn should_fallback_to_target(stats: &MtpStats) -> bool {
    if env_enabled("MTPLX_DISABLE_MTP_FALLBACK") {
        return false;
    }
    if stats.draft_tokens == 0 {
        return false;
    }
    let acceptance = f64::from(stats.accepted_draft_tokens) / f64::from(stats.draft_tokens);
    let full_cycle = stats.depth.max(1);
    if stats.draft_tokens >= full_cycle && stats.accepted_draft_tokens == 0 {
        return true;
    }
    (stats.draft_tokens >= full_cycle * 2 && acceptance < 0.35)
        || (stats.draft_tokens >= 12 && acceptance < 0.50)
}

#[cfg(feature = "native-mlx")]
fn adapt_mtp_depth(stats: &MtpStats, current_depth: u32) -> u32 {
    if current_depth <= 1 || env_enabled("MTPLX_DISABLE_MTP_ADAPTIVE_DEPTH") {
        return current_depth;
    }
    let index = current_depth as usize - 1;
    let drafted = stats.drafted_by_depth.get(index).copied().unwrap_or(0);
    if drafted < 4 {
        return current_depth;
    }
    let accepted = stats.accepted_by_depth.get(index).copied().unwrap_or(0);
    let acceptance = f64::from(accepted) / f64::from(drafted);
    if acceptance < 0.60 {
        current_depth - 1
    } else {
        current_depth
    }
}

#[cfg(feature = "native-mlx")]
fn generate_target_tail(
    session: &NativeMlxSession,
    request: &InferenceRequest,
    eos_token_ids: &[u32],
    sampling: crate::sampling::SamplingConfig,
    max_tokens: u32,
    mut completion_ids: Vec<u32>,
    mut state: crate::qwen36::DecodeState,
    mut logits: mlx_rs::Array,
    stats: MtpStats,
    mut stream: Option<&mut StreamState<'_>>,
) -> Result<GenerationOutput> {
    let mut finish_reason = FinishReason::Length;
    while completion_ids.len() < max_tokens as usize {
        let next = crate::sampling::distribution_from_logits(&logits, sampling)?.sample()?;
        if eos_token_ids.contains(&next) {
            finish_reason = FinishReason::Stop;
            break;
        }
        completion_ids.push(next);
        emit_stream_delta(&mut stream, &session.model, &completion_ids)?;
        if let Some(text) = stop_text_if_needed(&session.model, &completion_ids, &request.stop)? {
            finish_reason = FinishReason::Stop;
            return Ok(finalize_mtp_generation(
                completion_ids,
                finish_reason,
                Some(text),
                stats,
            ));
        }
        if completion_ids.len() >= max_tokens as usize {
            break;
        }
        let (next_logits, _next_hidden) =
            session
                .qwen
                .decode_step_logits_with_hidden(next, &session.plan, &mut state)?;
        logits = next_logits;
    }
    Ok(finalize_mtp_generation(
        completion_ids,
        finish_reason,
        None,
        stats,
    ))
}

#[cfg(feature = "native-mlx")]
fn append_mtp_history_from_commit(
    session: &NativeMlxSession,
    mtp_history_state: &mut crate::qwen36::MtpDecodeState,
    start_hidden: &mlx_rs::Array,
    committed: &[u32],
    committed_hidden: &mlx_rs::Array,
) -> Result<()> {
    for (index, token) in committed.iter().copied().enumerate() {
        let source_hidden = if index == 0 {
            start_hidden.clone()
        } else {
            let row = index as i32 - 1;
            committed_hidden.index((.., row..row + 1, ..))
        };
        let (_logits, _hidden) =
            session
                .qwen
                .mtp_draft_step(token, &source_hidden, &session.plan, mtp_history_state)?;
    }
    Ok(())
}

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
fn finalize_mtp_generation(
    completion_ids: Vec<u32>,
    finish_reason: FinishReason,
    stopped_text: Option<String>,
    mut stats: MtpStats,
) -> GenerationOutput {
    stats.acceptance_rate = if stats.draft_tokens == 0 {
        0.0
    } else {
        f64::from(stats.accepted_draft_tokens) / f64::from(stats.draft_tokens)
    };
    stats.acceptance_by_depth = stats
        .accepted_by_depth
        .iter()
        .zip(stats.drafted_by_depth.iter())
        .map(|(accepted, drafted)| {
            (*drafted > 0).then(|| f64::from(*accepted) / f64::from(*drafted))
        })
        .collect();
    stats.generated_tokens_per_cycle = if stats.cycles == 0 {
        0.0
    } else {
        completion_ids.len() as f64 / f64::from(stats.cycles)
    };
    GenerationOutput {
        completion_ids,
        finish_reason,
        stopped_text,
        mtp: Some(stats),
    }
}

#[cfg(feature = "native-mlx")]
fn prepare_prompt_state(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    prompt_ids: &[u32],
) -> Result<PreparedPrompt> {
    let started = Instant::now();
    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
    let needs_mtp_history =
        persistent_mtp && request.mtp && request.depth > 0 && session.qwen.mtp.is_some();

    if env_flag("MTPLX_PREFIX_CACHE", true) {
        if let Some(cached) = best_prompt_cache_match(session, prompt_ids) {
            let mut state = cached.state.clone();
            let mut logits = cached.logits.clone();
            let mut hidden = cached.hidden.clone();
            let mut mtp_history_state = if needs_mtp_history {
                cached.mtp_history_state.clone()
            } else {
                session.qwen.new_mtp_decode_state().unwrap_or_default()
            };
            let suffix = &prompt_ids[cached.prompt_ids.len()..];
            if !suffix.is_empty() {
                let base_hidden = hidden.clone();
                let (suffix_logits, suffix_hidden) = session
                    .qwen
                    .decode_tokens_logits_with_hidden(suffix, &session.plan, &mut state)?;
                let last = suffix.len() as i32 - 1;
                logits = suffix_logits.index((.., last..last + 1, ..));
                hidden = suffix_hidden.index((.., last..last + 1, ..));
                if request.profile_timings {
                    logits.eval()?;
                    hidden.eval()?;
                }
                if needs_mtp_history {
                    append_mtp_history_from_commit(
                        session,
                        &mut mtp_history_state,
                        &base_hidden,
                        suffix,
                        &suffix_hidden,
                    )?;
                }
            }
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
                prefix_cache_tokens: cached.prompt_ids.len(),
            });
        }
    }

    let prompt_i32 = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
    let mut state = session.qwen.new_decode_state(&session.plan)?;
    let (logits, hidden, prompt_hidden) = session.qwen.prefill_decode_state_with_hidden_sequence(
        &input_ids,
        &session.plan,
        &mut state,
    )?;
    if request.profile_timings {
        logits.eval()?;
        hidden.eval()?;
        prompt_hidden.eval()?;
    }
    let mtp_history_state = if needs_mtp_history {
        seed_mtp_history_from_prompt(session, prompt_ids, &prompt_hidden)?
    } else {
        session.qwen.new_mtp_decode_state().unwrap_or_default()
    };
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

#[cfg(feature = "native-mlx")]
fn is_token_prefix(prefix: &[u32], full: &[u32]) -> bool {
    prefix.len() <= full.len() && full.starts_with(prefix)
}

#[cfg(feature = "native-mlx")]
fn best_prompt_cache_match(
    session: &NativeMlxSession,
    prompt_ids: &[u32],
) -> Option<PromptStateCache> {
    session
        .prompt_caches
        .iter()
        .filter(|cached| is_token_prefix(&cached.prompt_ids, prompt_ids))
        .max_by_key(|cached| cached.prompt_ids.len())
        .cloned()
}

#[cfg(feature = "native-mlx")]
fn maybe_store_prompt_cache(
    session: &mut NativeMlxSession,
    prompt_ids: &[u32],
    state: &crate::qwen36::DecodeState,
    logits: &mlx_rs::Array,
    hidden: &mlx_rs::Array,
    mtp_history_state: &crate::qwen36::MtpDecodeState,
) {
    if !env_flag("MTPLX_PREFIX_CACHE", true) {
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
    });
    while session.prompt_caches.len() > max_entries {
        session.prompt_caches.remove(0);
    }
}

#[cfg(feature = "native-mlx")]
fn maybe_store_post_generation_cache(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    prompt_ids: &[u32],
    completion_ids: &[u32],
    mut state: crate::qwen36::DecodeState,
    start_hidden: mlx_rs::Array,
    mut mtp_history_state: crate::qwen36::MtpDecodeState,
) -> Result<()> {
    if !env_flag("MTPLX_POST_GENERATION_CACHE", true)
        || !env_flag("MTPLX_PREFIX_CACHE", true)
        || completion_ids.is_empty()
    {
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

#[cfg(feature = "native-mlx")]
fn maybe_store_chat_post_generation_cache(
    session: &mut NativeMlxSession,
    request: &InferenceRequest,
    assistant_text: &str,
) -> Result<()> {
    if !env_flag("MTPLX_CHAT_POST_GENERATION_CACHE", true)
        || !env_flag("MTPLX_PREFIX_CACHE", true)
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
    let prompt_i32 = prefix_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
    let mut state = session.qwen.new_decode_state(&session.plan)?;
    let (logits, hidden, prompt_hidden) = session.qwen.prefill_decode_state_with_hidden_sequence(
        &input_ids,
        &session.plan,
        &mut state,
    )?;
    if request.profile_timings {
        logits.eval()?;
        hidden.eval()?;
        prompt_hidden.eval()?;
    }
    let persistent_mtp = env_flag("MTPLX_PERSISTENT_MTP", true);
    let mtp_history_state =
        if persistent_mtp && request.mtp && request.depth > 0 && session.qwen.mtp.is_some() {
            seed_mtp_history_from_prompt(session, &prefix_ids, &prompt_hidden)?
        } else {
            session.qwen.new_mtp_decode_state().unwrap_or_default()
        };
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

#[cfg(feature = "native-mlx")]
fn prefix_cache_max_tokens() -> usize {
    ferrite_env_var("MTPLX_PREFIX_CACHE_MAX_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(65_536)
}

#[cfg(feature = "native-mlx")]
fn prefix_cache_max_entries() -> usize {
    ferrite_env_var("MTPLX_PREFIX_CACHE_ENTRIES")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(4)
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
            let prompt_ids = session.model.encode_prompt(request)?;
            let eos_token_ids = session.model.eos_token_ids();
            let sampling = crate::sampling::SamplingConfig {
                temperature: request.temperature,
                top_p: request.top_p,
                top_k: request.top_k,
            };
            let mut completion_ids = Vec::new();
            let mut finish_reason = FinishReason::Length;
            let max_tokens = request.max_tokens.unwrap_or(1);
            let mut prefill_s = 0.0;
            let mut prefix_cache_hit = false;
            let mut prefix_cache_tokens = 0_usize;
            let mut post_generation_cache_base = None;
            let decode_started = Instant::now();
            if max_tokens > 0 {
                let prepared = prepare_prompt_state(session, request, &prompt_ids)?;
                let mut state = prepared.state;
                let logits = prepared.logits;
                let hidden = prepared.hidden;
                let mtp_history_state = prepared.mtp_history_state;
                prefill_s = prepared.prefill_s;
                prefix_cache_hit = prepared.prefix_cache_hit;
                prefix_cache_tokens = prepared.prefix_cache_tokens;
                if request.mtp && request.depth > 0 && session.qwen.mtp.is_some() {
                    let cache_base_state = state.clone();
                    let cache_base_hidden = hidden.clone();
                    let cache_base_mtp_history_state = mtp_history_state.clone();
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
                    if !stopped_on_text {
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
                let cache_base_state = state.clone();
                let cache_base_hidden = hidden.clone();
                let cache_base_mtp_history_state = mtp_history_state.clone();
                post_generation_cache_base = Some((
                    cache_base_state,
                    cache_base_hidden,
                    cache_base_mtp_history_state,
                ));
                let mut next = crate::sampling::next_from_logits(&logits, sampling)?;
                if eos_token_ids.contains(&next) {
                    let total_s = total_started.elapsed().as_secs_f64();
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
}
