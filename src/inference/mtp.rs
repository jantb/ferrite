use std::time::Instant;

use anyhow::Result;
use mlx_rs::ops::indexing::IndexOp;

use super::{
    FinishReason, InferenceRequest, MtpStats, NativeMlxSession, check_kill_switch, env_enabled,
    env_flag, stop_text_if_needed,
};

pub(super) struct GenerationOutput {
    pub(super) completion_ids: Vec<u32>,
    pub(super) finish_reason: FinishReason,
    pub(super) stopped_text: Option<String>,
    pub(super) mtp: Option<MtpStats>,
}

pub(super) struct StreamState<'a> {
    pub(super) emitted_text: String,
    pub(super) on_delta: &'a mut dyn FnMut(&str) -> Result<()>,
}

impl<'a> StreamState<'a> {
    pub(super) fn emit(
        &mut self,
        model: &crate::model::LoadedModel,
        completion_ids: &[u32],
    ) -> Result<()> {
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

pub(super) fn emit_stream_delta(
    stream: &mut Option<&mut StreamState<'_>>,
    model: &crate::model::LoadedModel,
    completion_ids: &[u32],
) -> Result<()> {
    if let Some(stream) = stream.as_deref_mut() {
        stream.emit(model, completion_ids)?;
    }
    Ok(())
}

enum DraftCandidate {
    Greedy(u32),
    Sampled {
        token: u32,
        distribution: crate::sampling::TokenDistribution,
    },
}

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

pub(super) fn generate_mtp(
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
        check_kill_switch("generate_mtp_loop")?;
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
        check_kill_switch("generate_target_tail_loop")?;
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

pub(super) fn append_mtp_history_from_commit(
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
