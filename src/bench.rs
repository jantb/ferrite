use std::time::Instant;

use anyhow::Result;
use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct NativeMicrobenchResult {
    pub model: String,
    pub prompt_tokens: usize,
    pub iterations: u32,
    pub warmup: u32,
    pub elapsed_s: f64,
    pub layer0_mlp_passes_per_s: f64,
    pub prompt_token_equivalent_per_s: f64,
    pub output_shape: Vec<i32>,
    pub first_full_attention_q_shape: Vec<i32>,
    pub first_full_attention_k_shape: Vec<i32>,
    pub first_full_attention_v_shape: Vec<i32>,
    pub first_full_attention_output_shape: Vec<i32>,
    pub first_full_attention_block_shape: Vec<i32>,
    pub first_full_attention_causal_rope_block_shape: Vec<i32>,
    pub first_linear_attention_qkv_shapes: Vec<Vec<i32>>,
    pub first_linear_attention_reference_block_shape: Vec<i32>,
    pub ablated_full_model_logits_shape: Vec<i32>,
}

#[derive(Clone, Debug, Serialize)]
pub struct DecodeBenchResult {
    pub model: String,
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
    pub prompt_tokens: usize,
    pub generated_tokens: u32,
    pub finish_reason: crate::inference::FinishReason,
    pub load_s: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub total_s: f64,
    pub tokens_per_s: f64,
    pub steady_state_tokens_per_s: Option<f64>,
    pub text: String,
    pub backend: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct InferPathBenchResult {
    pub model: String,
    pub prompt_tokens: u32,
    pub requested_tokens: u32,
    pub iterations: u32,
    pub warmup: u32,
    pub mtp: bool,
    pub depth: u32,
    pub load_s: f64,
    pub avg_prefill_s: f64,
    pub avg_decode_s: f64,
    pub avg_total_s: f64,
    pub prefill_tokens_per_s: f64,
    pub output_tokens_per_s: f64,
    pub completion_tokens: u32,
    pub backend: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct MtpBenchResult {
    pub model: String,
    pub prompt_tokens: usize,
    pub requested_tokens: u32,
    pub generated_tokens: u32,
    pub depth: u32,
    pub finish_reason: crate::inference::FinishReason,
    pub load_s: f64,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub total_s: f64,
    pub tokens_per_s: f64,
    pub draft_tokens: u32,
    pub accepted_draft_tokens: u32,
    pub rejected_draft_tokens: u32,
    pub acceptance_rate: f64,
    pub text: String,
    pub backend: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qmv4BenchResult {
    pub k: i32,
    pub n: i32,
    pub group_size: i32,
    pub iterations: u32,
    pub warmup: u32,
    pub cases: Vec<Qmv4BenchCase>,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qmv4BenchCase {
    pub m: i32,
    pub stock_ms_per_iter: f64,
    pub ferrite_ms_per_iter: f64,
    pub speedup: f64,
    pub max_abs_diff: f32,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextProbeResult {
    pub model: String,
    pub source_prompt_tokens: usize,
    pub generate_tokens: u32,
    pub contexts: Vec<ContextProbeCase>,
    pub backend: String,
}

#[derive(Clone, Debug, Serialize)]
pub struct ContextProbeCase {
    pub requested_prompt_tokens: usize,
    pub prompt_tokens: usize,
    pub generated_tokens: u32,
    pub prefill_s: f64,
    pub decode_s: f64,
    pub total_s: f64,
    pub prefill_tokens_per_s: f64,
    pub output_tokens_per_s: f64,
    pub estimated_full_attention_kv_gib: f64,
    pub estimated_prompt_hidden_gib: f64,
}

#[cfg(feature = "native-mlx")]
pub fn run_qmv4_bench(
    m_values: &[i32],
    k: i32,
    n: i32,
    group_size: i32,
    iterations: u32,
    warmup: u32,
) -> Result<Qmv4BenchResult> {
    anyhow::ensure!(k > 0 && n > 0, "K and N must be positive");
    anyhow::ensure!(
        k % 512 == 0,
        "K must be divisible by 512 for qmv4 fast path"
    );
    anyhow::ensure!(
        matches!(group_size, 32 | 64 | 128),
        "group size must be 32, 64, or 128"
    );
    anyhow::ensure!(
        k % group_size == 0,
        "K must be divisible by the quantization group size"
    );
    anyhow::ensure!(iterations > 0, "iterations must be positive");
    anyhow::ensure!(
        crate::metal_kernels::metal_is_available(),
        "Metal is not available"
    );

    let dense_w_values = (0..(n * k))
        .map(|idx| (((idx * 17 + 11) % 41) as f32 - 20.0) / 17.0)
        .collect::<Vec<_>>();
    let dense_w =
        mlx_rs::Array::from_slice(&dense_w_values, &[n, k]).as_dtype(mlx_rs::Dtype::Bfloat16)?;
    let (weight, scales, biases) = mlx_rs::ops::quantize(&dense_w, group_size, 4)?;
    weight.eval()?;
    scales.eval()?;
    biases.eval()?;
    let linear = crate::mlx_backend::QuantizedLinear {
        weight,
        scales,
        biases,
        bias: None,
        group_size,
        bits: 4,
    };

    let mut cases = Vec::with_capacity(m_values.len());
    for &m in m_values {
        anyhow::ensure!((1..=6).contains(&m), "M must be in 1..=6 for qmv4");
        let x_values = (0..(m * k))
            .map(|idx| (((idx * 13 + 7) % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let x = mlx_rs::Array::from_slice(&x_values, &[m, k]).as_dtype(mlx_rs::Dtype::Bfloat16)?;

        for _ in 0..warmup {
            let stock = mlx_rs::ops::quantized_matmul(
                &x,
                &linear.weight,
                &linear.scales,
                &linear.biases,
                true,
                group_size,
                4,
            )?;
            stock.eval()?;
            let ferrite = crate::metal_kernels::small_m_qmv4_matmul_for_bench(&x, &linear)?
                .ok_or_else(|| anyhow::anyhow!("qmv4 fast path was not eligible"))?;
            ferrite.eval()?;
        }

        let stock_started = Instant::now();
        let mut stock_last = None;
        for _ in 0..iterations {
            let y = mlx_rs::ops::quantized_matmul(
                &x,
                &linear.weight,
                &linear.scales,
                &linear.biases,
                true,
                group_size,
                4,
            )?;
            y.eval()?;
            stock_last = Some(y);
        }
        let stock_elapsed_s = stock_started.elapsed().as_secs_f64();

        let ferrite_started = Instant::now();
        let mut ferrite_last = None;
        for _ in 0..iterations {
            let y = crate::metal_kernels::small_m_qmv4_matmul_for_bench(&x, &linear)?
                .ok_or_else(|| anyhow::anyhow!("qmv4 fast path was not eligible"))?;
            y.eval()?;
            ferrite_last = Some(y);
        }
        let ferrite_elapsed_s = ferrite_started.elapsed().as_secs_f64();

        let stock = stock_last
            .expect("stock qmv bench ran at least one iteration")
            .as_dtype(mlx_rs::Dtype::Float32)?;
        let ferrite = ferrite_last
            .expect("ferrite qmv bench ran at least one iteration")
            .as_dtype(mlx_rs::Dtype::Float32)?;
        stock.eval()?;
        ferrite.eval()?;
        let max_abs_diff = stock
            .as_slice::<f32>()
            .iter()
            .zip(ferrite.as_slice::<f32>().iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);

        let stock_ms_per_iter = stock_elapsed_s * 1000.0 / f64::from(iterations);
        let ferrite_ms_per_iter = ferrite_elapsed_s * 1000.0 / f64::from(iterations);
        cases.push(Qmv4BenchCase {
            m,
            stock_ms_per_iter,
            ferrite_ms_per_iter,
            speedup: stock_ms_per_iter / ferrite_ms_per_iter.max(f64::EPSILON),
            max_abs_diff,
        });
    }

    Ok(Qmv4BenchResult {
        k,
        n,
        group_size,
        iterations,
        warmup,
        cases,
    })
}

#[cfg(feature = "native-mlx")]
pub fn run_native_microbench(
    model_ref: &str,
    prompt: &str,
    iterations: u32,
    warmup: u32,
) -> Result<NativeMicrobenchResult> {
    let model = crate::model::LoadedModel::load(model_ref)?;
    let request = crate::inference::InferenceRequest {
        model: model_ref.to_string(),
        prompt: prompt.to_string(),
        system: None,
        messages: Vec::new(),
        stop: Vec::new(),
        max_tokens: Some(1),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        depth: 0,
        mtp: false,
        requested_context_tokens: None,
        profile_timings: false,
    };
    let prompt_ids = model.encode_prompt(&request)?;
    let prompt_i32 = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
    let weights = crate::mlx_backend::MlxWeightStore::load_model_dir(&model.path, &model.tensors)?;
    let qwen = crate::qwen36::Qwen36Weights::from_loaded(&model, &weights)?;
    let plan = crate::qwen36::Qwen36Plan::from_model(&model)?;

    let passes = iterations.max(1);
    for _ in 0..warmup {
        let out = layer0_mlp_pass(&qwen, &input_ids)?;
        out.eval()?;
    }

    let started = Instant::now();
    let mut output_shape = Vec::new();
    for _ in 0..passes {
        let out = layer0_mlp_pass(&qwen, &input_ids)?;
        out.eval()?;
        output_shape = out.shape().to_vec();
    }
    let elapsed_s = started.elapsed().as_secs_f64();
    let full_projection = first_full_attention_projection(&qwen, &plan, &input_ids)?;
    full_projection.q.eval()?;
    full_projection.k.eval()?;
    full_projection.v.eval()?;
    let first_full_attention_q_shape = full_projection.q.shape().to_vec();
    let first_full_attention_k_shape = full_projection.k.shape().to_vec();
    let first_full_attention_v_shape = full_projection.v.shape().to_vec();
    let full_output = first_full_attention_unmasked_no_rope(&qwen, &plan, &input_ids)?;
    full_output.eval()?;
    let first_full_attention_output_shape = full_output.shape().to_vec();
    let full_block = first_full_attention_block_no_rope(&qwen, &plan, &input_ids)?;
    full_block.eval()?;
    let first_full_attention_block_shape = full_block.shape().to_vec();
    let full_causal_rope_block = first_full_attention_block_causal_rope(&qwen, &plan, &input_ids)?;
    full_causal_rope_block.eval()?;
    let first_full_attention_causal_rope_block_shape = full_causal_rope_block.shape().to_vec();
    let linear_projection = first_linear_attention_projection(&qwen, &plan, &input_ids)?;
    linear_projection.q.eval()?;
    linear_projection.k.eval()?;
    linear_projection.v.eval()?;
    linear_projection.a.eval()?;
    linear_projection.b.eval()?;
    linear_projection.z.eval()?;
    let first_linear_attention_qkv_shapes = vec![
        linear_projection.q.shape().to_vec(),
        linear_projection.k.shape().to_vec(),
        linear_projection.v.shape().to_vec(),
        linear_projection.a.shape().to_vec(),
        linear_projection.b.shape().to_vec(),
        linear_projection.z.shape().to_vec(),
    ];
    let linear_reference_block = first_linear_attention_reference_block(&qwen, &plan, &input_ids)?;
    linear_reference_block.eval()?;
    let first_linear_attention_reference_block_shape = linear_reference_block.shape().to_vec();
    let ablated_logits = qwen.forward_ablate_linear_attention(&input_ids, &plan)?;
    ablated_logits.eval()?;
    let ablated_full_model_logits_shape = ablated_logits.shape().to_vec();

    Ok(NativeMicrobenchResult {
        model: model.path.display().to_string(),
        prompt_tokens: prompt_ids.len(),
        iterations: passes,
        warmup,
        elapsed_s,
        layer0_mlp_passes_per_s: f64::from(passes) / elapsed_s.max(f64::EPSILON),
        prompt_token_equivalent_per_s: (prompt_ids.len() as f64 * f64::from(passes))
            / elapsed_s.max(f64::EPSILON),
        output_shape,
        first_full_attention_q_shape,
        first_full_attention_k_shape,
        first_full_attention_v_shape,
        first_full_attention_output_shape,
        first_full_attention_block_shape,
        first_full_attention_causal_rope_block_shape,
        first_linear_attention_qkv_shapes,
        first_linear_attention_reference_block_shape,
        ablated_full_model_logits_shape,
    })
}

#[cfg(feature = "native-mlx")]
pub fn run_decode_bench(
    model_ref: &str,
    prompt: &str,
    tokens: u32,
    sampling: crate::sampling::SamplingConfig,
) -> Result<DecodeBenchResult> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model = crate::model::LoadedModel::load(model_ref)?;
    let request = crate::inference::InferenceRequest {
        model: model_ref.to_string(),
        prompt: prompt.to_string(),
        system: None,
        messages: Vec::new(),
        stop: Vec::new(),
        max_tokens: Some(tokens),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        depth: 0,
        mtp: false,
        requested_context_tokens: None,
        profile_timings: false,
    };
    let prompt_ids = model.encode_prompt(&request)?;
    let eos_token_ids = model.eos_token_ids();
    let weights = crate::mlx_backend::MlxWeightStore::load_model_dir(&model.path, &model.tensors)?;
    let qwen = crate::qwen36::Qwen36Weights::from_loaded(&model, &weights)?;
    let plan = crate::qwen36::Qwen36Plan::from_model(&model)?;
    let load_s = load_started.elapsed().as_secs_f64();

    let mut generated = Vec::with_capacity(tokens as usize);
    let decode_started = Instant::now();
    let prompt_i32 = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
    let mut state = qwen.new_decode_state(&plan)?;
    let prefill_started = Instant::now();
    let logits = qwen.prefill_decode_state(&input_ids, &plan, &mut state)?;
    let prefill_s = prefill_started.elapsed().as_secs_f64();
    let mut finish_reason = crate::inference::FinishReason::Length;
    if tokens > 0 {
        let mut next = crate::sampling::next_from_logits(&logits, sampling)?;
        if eos_token_ids.contains(&next) {
            let decode_s = decode_started.elapsed().as_secs_f64();
            let total_s = total_started.elapsed().as_secs_f64();
            return Ok(DecodeBenchResult {
                model: model.path.display().to_string(),
                temperature: sampling.temperature,
                top_p: sampling.top_p,
                top_k: sampling.top_k,
                prompt_tokens: prompt_ids.len(),
                generated_tokens: 0,
                finish_reason: crate::inference::FinishReason::Stop,
                load_s,
                prefill_s,
                decode_s,
                total_s,
                tokens_per_s: 0.0,
                steady_state_tokens_per_s: None,
                text: String::new(),
                backend: "native-mlx-rust-metal-gdn-cached".to_string(),
            });
        }
        generated.push(next);
        let step_started = Instant::now();
        for _ in 1..tokens {
            let logits = qwen.decode_step_logits(next, &plan, &mut state)?;
            next = crate::sampling::next_from_logits(&logits, sampling)?;
            if eos_token_ids.contains(&next) {
                finish_reason = crate::inference::FinishReason::Stop;
                break;
            }
            generated.push(next);
        }
        let step_s = step_started.elapsed().as_secs_f64();
        let decode_s = decode_started.elapsed().as_secs_f64();
        let text = model
            .tokenizer
            .decode(&generated, true)
            .map_err(|err| anyhow::anyhow!("decode completion: {err}"))?;
        let total_s = total_started.elapsed().as_secs_f64();

        return Ok(DecodeBenchResult {
            model: model.path.display().to_string(),
            temperature: sampling.temperature,
            top_p: sampling.top_p,
            top_k: sampling.top_k,
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated.len() as u32,
            finish_reason,
            load_s,
            prefill_s,
            decode_s,
            total_s,
            tokens_per_s: generated.len() as f64 / decode_s.max(f64::EPSILON),
            steady_state_tokens_per_s: if generated.len() > 1 {
                Some(generated.len().saturating_sub(1) as f64 / step_s.max(f64::EPSILON))
            } else {
                None
            },
            text,
            backend: "native-mlx-rust-metal-gdn-cached".to_string(),
        });
    }
    let decode_s = decode_started.elapsed().as_secs_f64();
    let text = model
        .tokenizer
        .decode(&generated, true)
        .map_err(|err| anyhow::anyhow!("decode completion: {err}"))?;
    let total_s = total_started.elapsed().as_secs_f64();

    Ok(DecodeBenchResult {
        model: model.path.display().to_string(),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        prompt_tokens: prompt_ids.len(),
        generated_tokens: generated.len() as u32,
        finish_reason,
        load_s,
        prefill_s,
        decode_s,
        total_s,
        tokens_per_s: generated.len() as f64 / decode_s.max(f64::EPSILON),
        steady_state_tokens_per_s: None,
        text,
        backend: "native-mlx-rust-metal-gdn-cached".to_string(),
    })
}

#[cfg(feature = "native-mlx")]
pub fn run_infer_path_bench(
    model_ref: &str,
    prompt: &str,
    tokens: u32,
    sampling: crate::sampling::SamplingConfig,
    mtp: bool,
    depth: u32,
    iterations: u32,
    warmup: u32,
    profile_timings: bool,
) -> Result<InferPathBenchResult> {
    use crate::inference::InferenceBackend;

    let backend = crate::inference::NativeMlxBackend::new();
    let load_s = backend.preload(model_ref)?;
    let request = crate::inference::InferenceRequest {
        model: model_ref.to_string(),
        prompt: prompt.to_string(),
        system: None,
        messages: Vec::new(),
        stop: Vec::new(),
        max_tokens: Some(tokens),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        depth,
        mtp,
        requested_context_tokens: None,
        profile_timings,
    };

    for _ in 0..warmup {
        let _ = backend.infer(&request)?;
    }

    let passes = iterations.max(1);
    let mut prompt_tokens = 0_u32;
    let mut completion_tokens = 0_u32;
    let mut prefill_s = 0.0_f64;
    let mut decode_s = 0.0_f64;
    let mut total_s = 0.0_f64;
    let mut backend_name = String::new();

    for _ in 0..passes {
        let response = backend.infer(&request)?;
        prompt_tokens = response.prompt_tokens;
        completion_tokens = response.completion_tokens;
        backend_name = response.backend;
        if let Some(timings) = response.timings {
            prefill_s += timings.prefill_s;
            decode_s += timings.decode_s;
            total_s += timings.total_s;
        }
    }

    let avg_prefill_s = prefill_s / f64::from(passes);
    let avg_decode_s = decode_s / f64::from(passes);
    let avg_total_s = total_s / f64::from(passes);
    Ok(InferPathBenchResult {
        model: model_ref.to_string(),
        prompt_tokens,
        requested_tokens: tokens,
        iterations: passes,
        warmup,
        mtp,
        depth,
        load_s,
        avg_prefill_s,
        avg_decode_s,
        avg_total_s,
        prefill_tokens_per_s: f64::from(prompt_tokens) / avg_prefill_s.max(f64::EPSILON),
        output_tokens_per_s: f64::from(completion_tokens) / avg_decode_s.max(f64::EPSILON),
        completion_tokens,
        backend: backend_name,
    })
}

#[cfg(feature = "native-mlx")]
pub fn run_mtp_bench(
    model_ref: &str,
    prompt: &str,
    tokens: u32,
    depth: u32,
) -> Result<MtpBenchResult> {
    let total_started = Instant::now();
    let load_started = Instant::now();
    let model = crate::model::LoadedModel::load(model_ref)?;
    let request = crate::inference::InferenceRequest {
        model: model_ref.to_string(),
        prompt: prompt.to_string(),
        system: None,
        messages: Vec::new(),
        stop: Vec::new(),
        max_tokens: Some(tokens),
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
        depth,
        mtp: true,
        requested_context_tokens: None,
        profile_timings: false,
    };
    let prompt_ids = model.encode_prompt(&request)?;
    let eos_token_ids = model.eos_token_ids();
    let weights = crate::mlx_backend::MlxWeightStore::load_model_dir(&model.path, &model.tensors)?;
    let qwen = crate::qwen36::Qwen36Weights::from_loaded(&model, &weights)?;
    if qwen.mtp.is_none() {
        anyhow::bail!("model has no MTP sidecar tensors");
    }
    let plan = crate::qwen36::Qwen36Plan::from_model(&model)?;
    let load_s = load_started.elapsed().as_secs_f64();

    let greedy = crate::sampling::SamplingConfig {
        temperature: 0.0,
        top_p: 1.0,
        top_k: 1,
    };
    let mut generated = Vec::with_capacity(tokens as usize);
    let mut finish_reason = crate::inference::FinishReason::Length;
    let mut draft_tokens = 0_u32;
    let mut accepted_draft_tokens = 0_u32;
    let mut rejected_draft_tokens = 0_u32;

    let decode_started = Instant::now();
    let prompt_i32 = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
    let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
    let mut state = qwen.new_decode_state(&plan)?;
    let prefill_started = Instant::now();
    let (mut logits, mut hidden) =
        qwen.prefill_decode_state_with_hidden(&input_ids, &plan, &mut state)?;
    let prefill_s = prefill_started.elapsed().as_secs_f64();

    while generated.len() < tokens as usize {
        let primary = crate::sampling::next_from_logits(&logits, greedy)?;
        if eos_token_ids.contains(&primary) {
            finish_reason = crate::inference::FinishReason::Stop;
            break;
        }

        let base_state = state.clone();
        let base_logits = logits.clone();
        let base_hidden = hidden.clone();
        let mut verify_state = base_state.clone();
        let (mut verified_logits, mut verified_hidden) =
            qwen.decode_step_logits_with_hidden(primary, &plan, &mut verify_state)?;

        let mut mtp_state = qwen.new_mtp_decode_state()?;
        let mut draft_hidden = base_hidden;
        let mut draft_input = primary;
        let mut drafts = Vec::new();
        for _ in 0..depth {
            if generated.len() + 1 + drafts.len() >= tokens as usize {
                break;
            }
            let (draft_logits, next_draft_hidden) =
                qwen.mtp_draft_step(draft_input, &draft_hidden, &plan, &mut mtp_state)?;
            let draft = crate::sampling::next_from_logits(&draft_logits, greedy)?;
            drafts.push(draft);
            draft_tokens += 1;
            draft_input = draft;
            draft_hidden = next_draft_hidden;
        }

        let mut accepted = Vec::new();
        let mut correction = None;
        let mut stop_after_accepted = false;
        for draft in drafts {
            let target = crate::sampling::next_from_logits(&verified_logits, greedy)?;
            if eos_token_ids.contains(&target) {
                rejected_draft_tokens += 1;
                finish_reason = crate::inference::FinishReason::Stop;
                stop_after_accepted = true;
                break;
            }
            if draft == target {
                accepted.push(draft);
                accepted_draft_tokens += 1;
                let (next_logits, next_hidden) =
                    qwen.decode_step_logits_with_hidden(draft, &plan, &mut verify_state)?;
                verified_logits = next_logits;
                verified_hidden = next_hidden;
            } else {
                rejected_draft_tokens += 1;
                correction = Some(target);
                break;
            }
        }

        if correction.is_none() && !stop_after_accepted {
            generated.push(primary);
            for token in accepted {
                if generated.len() >= tokens as usize {
                    break;
                }
                generated.push(token);
            }
            state = verify_state;
            logits = verified_logits;
            hidden = verified_hidden;
            continue;
        }

        state = base_state;
        logits = base_logits;
        let mut committed = Vec::with_capacity(accepted.len() + 2);
        committed.push(primary);
        committed.extend(accepted);
        if let Some(token) = correction {
            if generated.len() + committed.len() < tokens as usize {
                committed.push(token);
            }
        }
        for token in committed {
            let (next_logits, next_hidden) =
                qwen.decode_step_logits_with_hidden(token, &plan, &mut state)?;
            generated.push(token);
            logits = next_logits;
            hidden = next_hidden;
            if generated.len() >= tokens as usize {
                break;
            }
        }
        if stop_after_accepted {
            break;
        }
    }

    let decode_s = decode_started.elapsed().as_secs_f64();
    let text = model
        .tokenizer
        .decode(&generated, true)
        .map_err(|err| anyhow::anyhow!("decode completion: {err}"))?;
    let total_s = total_started.elapsed().as_secs_f64();
    let acceptance_rate = if draft_tokens == 0 {
        0.0
    } else {
        f64::from(accepted_draft_tokens) / f64::from(draft_tokens)
    };

    Ok(MtpBenchResult {
        model: model.path.display().to_string(),
        prompt_tokens: prompt_ids.len(),
        requested_tokens: tokens,
        generated_tokens: generated.len() as u32,
        depth,
        finish_reason,
        load_s,
        prefill_s,
        decode_s,
        total_s,
        tokens_per_s: generated.len() as f64 / decode_s.max(f64::EPSILON),
        draft_tokens,
        accepted_draft_tokens,
        rejected_draft_tokens,
        acceptance_rate,
        text,
        backend: "native-mlx-rust-mtp-greedy-verified".to_string(),
    })
}

#[cfg(feature = "native-mlx")]
pub fn run_context_probe(
    model_ref: &str,
    prompt: &str,
    context_tokens: &[usize],
    generate_tokens: u32,
    sampling: crate::sampling::SamplingConfig,
) -> Result<ContextProbeResult> {
    let load_started = Instant::now();
    let model = crate::model::LoadedModel::load(model_ref)?;
    let seed_request = crate::inference::InferenceRequest {
        model: model_ref.to_string(),
        prompt: prompt.to_string(),
        system: None,
        messages: Vec::new(),
        stop: Vec::new(),
        max_tokens: Some(generate_tokens),
        temperature: sampling.temperature,
        top_p: sampling.top_p,
        top_k: sampling.top_k,
        depth: 0,
        mtp: false,
        requested_context_tokens: None,
        profile_timings: false,
    };
    let source_prompt_ids = model.encode_prompt(&seed_request)?;
    if source_prompt_ids.is_empty() {
        anyhow::bail!("prompt produced no tokens");
    }
    let weights = crate::mlx_backend::MlxWeightStore::load_model_dir(&model.path, &model.tensors)?;
    let qwen = crate::qwen36::Qwen36Weights::from_loaded(&model, &weights)?;
    let plan = crate::qwen36::Qwen36Plan::from_model(&model)?;
    let eos_token_ids = model.eos_token_ids();
    let _load_s = load_started.elapsed().as_secs_f64();

    let mut contexts = Vec::with_capacity(context_tokens.len());
    for target_tokens in context_tokens {
        let prompt_ids = repeat_tokens_to_len(&source_prompt_ids, *target_tokens);
        let total_started = Instant::now();
        let prompt_i32 = prompt_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
        let input_ids = mlx_rs::Array::from_slice(&prompt_i32, &[1, prompt_i32.len() as i32]);
        let mut state = qwen.new_decode_state(&plan)?;

        let prefill_started = Instant::now();
        let logits = qwen.prefill_decode_state(&input_ids, &plan, &mut state)?;
        logits.eval()?;
        let prefill_s = prefill_started.elapsed().as_secs_f64();

        let decode_started = Instant::now();
        let mut generated = 0_u32;
        if generate_tokens > 0 {
            let mut next = crate::sampling::next_from_logits(&logits, sampling)?;
            if !eos_token_ids.contains(&next) {
                generated += 1;
                for _ in 1..generate_tokens {
                    let logits = qwen.decode_step_logits(next, &plan, &mut state)?;
                    logits.eval()?;
                    next = crate::sampling::next_from_logits(&logits, sampling)?;
                    if eos_token_ids.contains(&next) {
                        break;
                    }
                    generated += 1;
                }
            }
        }
        let decode_s = decode_started.elapsed().as_secs_f64();
        let total_s = total_started.elapsed().as_secs_f64();
        contexts.push(ContextProbeCase {
            requested_prompt_tokens: *target_tokens,
            prompt_tokens: prompt_ids.len(),
            generated_tokens: generated,
            prefill_s,
            decode_s,
            total_s,
            prefill_tokens_per_s: prompt_ids.len() as f64 / prefill_s.max(f64::EPSILON),
            output_tokens_per_s: generated as f64 / decode_s.max(f64::EPSILON),
            estimated_full_attention_kv_gib: estimate_full_attention_kv_gib(
                &plan,
                prompt_ids.len(),
            ),
            estimated_prompt_hidden_gib: estimate_prompt_hidden_gib(&plan, prompt_ids.len()),
        });
    }

    Ok(ContextProbeResult {
        model: model.path.display().to_string(),
        source_prompt_tokens: source_prompt_ids.len(),
        generate_tokens,
        contexts,
        backend: "native-mlx-rust-context-probe".to_string(),
    })
}

fn repeat_tokens_to_len(source: &[u32], target_len: usize) -> Vec<u32> {
    let target_len = target_len.max(1);
    source.iter().copied().cycle().take(target_len).collect()
}

#[cfg(feature = "native-mlx")]
fn estimate_full_attention_kv_gib(plan: &crate::qwen36::Qwen36Plan, tokens: usize) -> f64 {
    let bytes = tokens as f64
        * plan.full_attention_layers.len() as f64
        * 2.0
        * plan.num_key_value_heads as f64
        * plan.head_dim as f64
        * 2.0;
    bytes / 1024.0 / 1024.0 / 1024.0
}

#[cfg(feature = "native-mlx")]
fn estimate_prompt_hidden_gib(plan: &crate::qwen36::Qwen36Plan, tokens: usize) -> f64 {
    let bytes = tokens as f64 * plan.hidden_size as f64 * 2.0;
    bytes / 1024.0 / 1024.0 / 1024.0
}

#[cfg(feature = "native-mlx")]
fn layer0_mlp_pass(
    qwen: &crate::qwen36::Qwen36Weights,
    input_ids: &mlx_rs::Array,
) -> Result<mlx_rs::Array> {
    let hidden = qwen.embeddings.forward(input_ids)?;
    let normed = qwen.layers[0].input_norm.forward(&hidden)?;
    qwen.layers[0].mlp.forward(&normed)
}

#[cfg(feature = "native-mlx")]
fn first_full_attention_projection(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<crate::qwen36::FullAttentionProjection> {
    let layer_index = *plan
        .full_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no full-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    let normed = qwen.layers[layer_index].input_norm.forward(&hidden)?;
    let crate::qwen36::AttentionWeights::Full(attn) = &qwen.layers[layer_index].attention else {
        anyhow::bail!("planned full-attention layer {layer_index} was not full attention");
    };
    attn.project(
        &normed,
        plan.num_attention_heads as i32,
        plan.num_key_value_heads as i32,
        plan.head_dim as i32,
    )
}

#[cfg(feature = "native-mlx")]
fn first_full_attention_unmasked_no_rope(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<mlx_rs::Array> {
    let layer_index = *plan
        .full_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no full-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    let normed = qwen.layers[layer_index].input_norm.forward(&hidden)?;
    let crate::qwen36::AttentionWeights::Full(attn) = &qwen.layers[layer_index].attention else {
        anyhow::bail!("planned full-attention layer {layer_index} was not full attention");
    };
    attn.forward_unmasked_no_rope(
        &normed,
        plan.num_attention_heads as i32,
        plan.num_key_value_heads as i32,
        plan.head_dim as i32,
    )
}

#[cfg(feature = "native-mlx")]
fn first_full_attention_block_no_rope(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<mlx_rs::Array> {
    let layer_index = *plan
        .full_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no full-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    qwen.layers[layer_index].forward_full_attention_no_rope(
        &hidden,
        plan.num_attention_heads as i32,
        plan.num_key_value_heads as i32,
        plan.head_dim as i32,
    )
}

#[cfg(feature = "native-mlx")]
fn first_full_attention_block_causal_rope(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<mlx_rs::Array> {
    let layer_index = *plan
        .full_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no full-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    qwen.layers[layer_index].forward_full_attention_causal_rope(
        &hidden,
        plan.num_attention_heads as i32,
        plan.num_key_value_heads as i32,
        plan.head_dim as i32,
        plan.rope_dimensions as i32,
        plan.rope_theta,
        0,
    )
}

#[cfg(feature = "native-mlx")]
fn first_linear_attention_projection(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<crate::qwen36::LinearAttentionProjection> {
    let layer_index = *plan
        .linear_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no linear-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    let normed = qwen.layers[layer_index].input_norm.forward(&hidden)?;
    let crate::qwen36::AttentionWeights::Linear(attn) = &qwen.layers[layer_index].attention else {
        anyhow::bail!("planned linear-attention layer {layer_index} was not linear attention");
    };
    attn.project(
        &normed,
        plan.linear_num_key_heads as i32,
        plan.linear_key_head_dim as i32,
        plan.linear_num_value_heads as i32,
        plan.linear_value_head_dim as i32,
    )
}

#[cfg(feature = "native-mlx")]
fn first_linear_attention_reference_block(
    qwen: &crate::qwen36::Qwen36Weights,
    plan: &crate::qwen36::Qwen36Plan,
    input_ids: &mlx_rs::Array,
) -> Result<mlx_rs::Array> {
    let layer_index = *plan
        .linear_attention_layers
        .first()
        .ok_or_else(|| anyhow::anyhow!("model has no linear-attention layers"))?
        as usize;
    let hidden = qwen.embeddings.forward(input_ids)?;
    qwen.layers[layer_index].forward_reference(&hidden, plan, 0)
}

#[cfg(not(feature = "native-mlx"))]
pub fn run_native_microbench(
    _model_ref: &str,
    _prompt: &str,
    _iterations: u32,
    _warmup: u32,
) -> Result<NativeMicrobenchResult> {
    anyhow::bail!("native-mlx feature is disabled")
}
