use std::sync::Arc;
use std::time::Instant;

use anyhow::{Result, bail};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;

use super::masks::decode_block_causal_mask;
use super::{
    DecodeProfileTimings, FullAttentionCache, LayerDecodeState, LinearAttentionCache, Qwen36Plan,
    array_to_f32_vec,
};

#[derive(Debug)]
pub struct LayerWeights {
    pub input_norm: crate::mlx_backend::RmsNorm,
    pub post_attention_norm: crate::mlx_backend::RmsNorm,
    pub attention: AttentionWeights,
    pub mlp: MlpWeights,
}

impl LayerWeights {
    fn forward_residual_mlp(
        &self,
        x: &mlx_rs::Array,
        attn_out: &mlx_rs::Array,
    ) -> Result<mlx_rs::Array> {
        let hidden = x + attn_out;
        let mlp_normed = self.post_attention_norm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_normed)?;
        Ok(&hidden + &mlp_out)
    }

    fn forward_residual_mlp_profiled(
        &self,
        x: &mlx_rs::Array,
        attn_out: &mlx_rs::Array,
        profile: &mut DecodeProfileTimings,
    ) -> Result<mlx_rs::Array> {
        let started = Instant::now();
        let hidden = x + attn_out;
        let mlp_normed = self.post_attention_norm.forward(&hidden)?;
        hidden.eval()?;
        mlp_normed.eval()?;
        profile.layer_glue_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let mlp_out = self.mlp.forward(&mlp_normed)?;
        mlp_out.eval()?;
        profile.mlp_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let output = &hidden + &mlp_out;
        output.eval()?;
        profile.layer_glue_s += started.elapsed().as_secs_f64();
        Ok(output)
    }

    pub fn forward_full_attention_no_rope(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<mlx_rs::Array> {
        let AttentionWeights::Full(attn) = &self.attention else {
            bail!("forward_full_attention_no_rope called on a linear-attention layer");
        };
        let normed = self.input_norm.forward(x)?;
        let attn_out = attn.forward_unmasked_no_rope(&normed, num_heads, num_kv_heads, head_dim)?;
        self.forward_residual_mlp(x, &attn_out)
    }

    pub fn forward_full_attention_causal_rope(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dimensions: i32,
        rope_theta: f32,
        offset: i32,
    ) -> Result<mlx_rs::Array> {
        let AttentionWeights::Full(attn) = &self.attention else {
            bail!("forward_full_attention_causal_rope called on a linear-attention layer");
        };
        let normed = self.input_norm.forward(x)?;
        let attn_out = attn.forward_causal_rope(
            &normed,
            num_heads,
            num_kv_heads,
            head_dim,
            rope_dimensions,
            rope_theta,
            offset,
        )?;
        self.forward_residual_mlp(x, &attn_out)
    }

    pub fn forward_with_linear_attention_ablation(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
    ) -> Result<mlx_rs::Array> {
        match &self.attention {
            AttentionWeights::Full(_) => self.forward_full_attention_causal_rope(
                x,
                plan.num_attention_heads as i32,
                plan.num_key_value_heads as i32,
                plan.head_dim as i32,
                plan.rope_dimensions as i32,
                plan.rope_theta,
                position_offset,
            ),
            AttentionWeights::Linear(_) => {
                let h = x.clone();
                let mlp_normed = self.post_attention_norm.forward(&h)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&h + &mlp_out)
            }
        }
    }

    pub fn forward_reference(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
    ) -> Result<mlx_rs::Array> {
        match &self.attention {
            AttentionWeights::Full(_) => self.forward_full_attention_causal_rope(
                x,
                plan.num_attention_heads as i32,
                plan.num_key_value_heads as i32,
                plan.head_dim as i32,
                plan.rope_dimensions as i32,
                plan.rope_theta,
                position_offset,
            ),
            AttentionWeights::Linear(attn) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference(&normed, plan)?;
                self.forward_residual_mlp(x, &attn_out)
            }
        }
    }

    pub fn forward_prefill(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
        cache: &mut LayerDecodeState,
    ) -> Result<mlx_rs::Array> {
        match (&self.attention, cache) {
            (AttentionWeights::Full(attn), LayerDecodeState::Full(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_prefill(
                    &normed,
                    plan.num_attention_heads as i32,
                    plan.num_key_value_heads as i32,
                    plan.head_dim as i32,
                    plan.rope_dimensions as i32,
                    plan.rope_theta,
                    position_offset,
                    cache,
                )?;
                self.forward_residual_mlp(x, &attn_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                self.forward_residual_mlp(x, &attn_out)
            }
            _ => bail!("decode state kind does not match layer kind"),
        }
    }

    pub fn forward_decode_step(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
        cache: &mut LayerDecodeState,
    ) -> Result<mlx_rs::Array> {
        match (&self.attention, cache) {
            (AttentionWeights::Full(attn), LayerDecodeState::Full(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_decode_step(
                    &normed,
                    plan.num_attention_heads as i32,
                    plan.num_key_value_heads as i32,
                    plan.head_dim as i32,
                    plan.rope_dimensions as i32,
                    plan.rope_theta,
                    position_offset,
                    cache,
                )?;
                self.forward_residual_mlp(x, &attn_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                self.forward_residual_mlp(x, &attn_out)
            }
            _ => bail!("decode state kind does not match layer kind"),
        }
    }

    pub fn forward_decode_tokens(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
        cache: &mut LayerDecodeState,
    ) -> Result<mlx_rs::Array> {
        match (&self.attention, cache) {
            (AttentionWeights::Full(attn), LayerDecodeState::Full(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_decode_tokens(
                    &normed,
                    plan.num_attention_heads as i32,
                    plan.num_key_value_heads as i32,
                    plan.head_dim as i32,
                    plan.rope_dimensions as i32,
                    plan.rope_theta,
                    position_offset,
                    cache,
                )?;
                self.forward_residual_mlp(x, &attn_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                self.forward_residual_mlp(x, &attn_out)
            }
            _ => bail!("decode state kind does not match layer kind"),
        }
    }

    pub fn forward_decode_tokens_profiled(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        position_offset: i32,
        cache: &mut LayerDecodeState,
        profile: &mut DecodeProfileTimings,
    ) -> Result<mlx_rs::Array> {
        match (&self.attention, cache) {
            (AttentionWeights::Full(attn), LayerDecodeState::Full(cache)) => {
                let started = Instant::now();
                let normed = self.input_norm.forward(x)?;
                normed.eval()?;
                profile.layer_glue_s += started.elapsed().as_secs_f64();

                let started = Instant::now();
                let attn_out = attn.forward_decode_tokens(
                    &normed,
                    plan.num_attention_heads as i32,
                    plan.num_key_value_heads as i32,
                    plan.head_dim as i32,
                    plan.rope_dimensions as i32,
                    plan.rope_theta,
                    position_offset,
                    cache,
                )?;
                attn_out.eval()?;
                profile.full_attention_s += started.elapsed().as_secs_f64();

                self.forward_residual_mlp_profiled(x, &attn_out, profile)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let started = Instant::now();
                let normed = self.input_norm.forward(x)?;
                normed.eval()?;
                profile.layer_glue_s += started.elapsed().as_secs_f64();

                let started = Instant::now();
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                attn_out.eval()?;
                profile.linear_attention_s += started.elapsed().as_secs_f64();

                self.forward_residual_mlp_profiled(x, &attn_out, profile)
            }
            _ => bail!("decode state kind does not match layer kind"),
        }
    }
}

#[derive(Debug)]
pub enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(LinearAttentionWeights),
}

#[derive(Debug)]
pub struct FullAttentionWeights {
    pub q_proj: crate::mlx_backend::QuantizedLinear,
    pub k_proj: crate::mlx_backend::QuantizedLinear,
    pub v_proj: crate::mlx_backend::QuantizedLinear,
    pub o_proj: crate::mlx_backend::QuantizedLinear,
    pub q_norm: crate::mlx_backend::RmsNorm,
    pub k_norm: crate::mlx_backend::RmsNorm,
}

pub struct FullAttentionProjection {
    pub q: mlx_rs::Array,
    pub k: mlx_rs::Array,
    pub v: mlx_rs::Array,
    pub gate: Option<mlx_rs::Array>,
}

impl FullAttentionWeights {
    pub fn project(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<FullAttentionProjection> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("full attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0];
        let tokens = shape[1];
        let q_raw = self.q_proj.forward(x)?;
        let q_shape = q_raw.shape();
        let (q, gate) = if q_shape.last().copied() == Some(num_heads * head_dim * 2) {
            // Qwen3.6 full-attention layers with attn_output_gate pack
            // [query, gate] per head in q_proj.
            let mut parts = q_raw
                .reshape(&[batch, tokens, num_heads, head_dim * 2])?
                .split(2, -1)?;
            let q = parts.remove(0);
            let gate = parts
                .remove(0)
                .reshape(&[batch, tokens, num_heads * head_dim])?;
            (q, Some(gate))
        } else {
            (q_raw.reshape(&[batch, tokens, num_heads, head_dim])?, None)
        };
        let q = self.q_norm.forward(&q)?.transpose_axes(&[0, 2, 1, 3])?;
        let k = self
            .k_norm
            .forward(
                &self
                    .k_proj
                    .forward(x)?
                    .reshape(&[batch, tokens, num_kv_heads, head_dim])?,
            )?
            .transpose_axes(&[0, 2, 1, 3])?;
        let v = self
            .v_proj
            .forward(x)?
            .reshape(&[batch, tokens, num_kv_heads, head_dim])?
            .transpose_axes(&[0, 2, 1, 3])?;
        Ok(FullAttentionProjection { q, k, v, gate })
    }

    pub fn forward_unmasked_no_rope(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("full attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0];
        let tokens = shape[1];
        let projected = self.project(x, num_heads, num_kv_heads, head_dim)?;
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &projected.q,
            &projected.k,
            &projected.v,
            1.0 / (head_dim as f32).sqrt(),
            None,
        )?;
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            tokens,
            num_heads * head_dim,
        ])?;
        let merged = apply_attention_gate(merged, projected.gate.as_ref())?;
        self.o_proj.forward(&merged)
    }

    pub fn forward_causal_rope(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dimensions: i32,
        rope_theta: f32,
        offset: i32,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("full attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0];
        let tokens = shape[1];
        let projected = self.project(x, num_heads, num_kv_heads, head_dim)?;
        let q = mlx_rs::fast::rope(
            &projected.q,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let k = mlx_rs::fast::rope(
            &projected.k,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &projected.v,
            1.0 / (head_dim as f32).sqrt(),
            mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
        )?;
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            tokens,
            num_heads * head_dim,
        ])?;
        let merged = apply_attention_gate(merged, projected.gate.as_ref())?;
        self.o_proj.forward(&merged)
    }

    pub fn forward_prefill(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dimensions: i32,
        rope_theta: f32,
        offset: i32,
        cache: &mut FullAttentionCache,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("full attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0];
        let tokens = shape[1];
        let projected = self.project(x, num_heads, num_kv_heads, head_dim)?;
        let q = mlx_rs::fast::rope(
            &projected.q,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let k = mlx_rs::fast::rope(
            &projected.k,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &k,
            &projected.v,
            1.0 / (head_dim as f32).sqrt(),
            mlx_rs::fast::ScaledDotProductAttentionMask::Causal,
        )?;
        let gate = projected.gate;
        cache.set_prefill(k, projected.v)?;
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            tokens,
            num_heads * head_dim,
        ])?;
        let merged = apply_attention_gate(merged, gate.as_ref())?;
        self.o_proj.forward(&merged)
    }

    pub fn forward_decode_step(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dimensions: i32,
        rope_theta: f32,
        offset: i32,
        cache: &mut FullAttentionCache,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 || shape[1] != 1 {
            bail!("full attention decode input must be [batch, 1, hidden], got {shape:?}");
        }
        let batch = shape[0];
        let projected = self.project(x, num_heads, num_kv_heads, head_dim)?;
        let q = mlx_rs::fast::rope(
            &projected.q,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let k = mlx_rs::fast::rope(
            &projected.k,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let (active_k, active_v, _) = cache.append_and_fetch(k, projected.v)?;
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            &active_k,
            &active_v,
            1.0 / (head_dim as f32).sqrt(),
            None,
        )?;
        let merged =
            attended
                .transpose_axes(&[0, 2, 1, 3])?
                .reshape(&[batch, 1, num_heads * head_dim])?;
        let merged = apply_attention_gate(merged, projected.gate.as_ref())?;
        self.o_proj.forward(&merged)
    }

    pub fn forward_decode_tokens(
        &self,
        x: &mlx_rs::Array,
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        rope_dimensions: i32,
        rope_theta: f32,
        offset: i32,
        cache: &mut FullAttentionCache,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] < 1 {
            bail!("full attention decode input must be [1, tokens, hidden], got {shape:?}");
        }
        if shape[1] == 1 {
            return self.forward_decode_step(
                x,
                num_heads,
                num_kv_heads,
                head_dim,
                rope_dimensions,
                rope_theta,
                offset,
                cache,
            );
        }
        let batch = shape[0];
        let tokens = shape[1];
        let projected = self.project(x, num_heads, num_kv_heads, head_dim)?;
        let q = mlx_rs::fast::rope(
            &projected.q,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let k = mlx_rs::fast::rope(
            &projected.k,
            rope_dimensions,
            false,
            rope_theta,
            1.0,
            offset,
            None,
        )?;
        let scale = 1.0 / (head_dim as f32).sqrt();
        let prev_len = cache.current_len();
        let attended = if blockwise_full_attention_enabled(tokens, prev_len) {
            let prev_len = cache.append_without_fetch(k, projected.v)?;
            let blocks = cache.active_block_slices(blockwise_full_attention_block_tokens())?;
            blockwise_decode_scaled_dot_product_attention(&q, &blocks, tokens, prev_len, scale)?
        } else {
            let (active_k, active_v, prev_len) = cache.append_and_fetch(k, projected.v)?;
            let total_keys = active_k.shape()[2];
            if let Some(chunk) = split_full_attention_chunk_tokens(tokens, prev_len) {
                split_decode_scaled_dot_product_attention(
                    &q, &active_k, &active_v, tokens, prev_len, chunk, scale,
                )?
            } else {
                let mask = decode_block_causal_mask(tokens, total_keys, prev_len, q.dtype())?;
                mlx_rs::fast::scaled_dot_product_attention(&q, &active_k, &active_v, scale, &mask)?
            }
        };
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            tokens,
            num_heads * head_dim,
        ])?;
        let merged = apply_attention_gate(merged, projected.gate.as_ref())?;
        self.o_proj.forward(&merged)
    }
}

fn blockwise_full_attention_enabled(tokens: i32, prev_len: i32) -> bool {
    let enabled = std::env::var("FERRITE_BLOCKWISE_FULL_ATTN")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false);
    if !enabled {
        return false;
    }
    let threshold = std::env::var("FERRITE_BLOCKWISE_FULL_ATTN_THRESHOLD")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|value| *value >= 0)
        .unwrap_or(1024);
    tokens > 1 && prev_len >= threshold
}

fn blockwise_full_attention_block_tokens() -> i32 {
    std::env::var("FERRITE_BLOCKWISE_FULL_ATTN_BLOCK_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(512)
}

fn blockwise_decode_scaled_dot_product_attention(
    q: &mlx_rs::Array,
    blocks: &[(i32, mlx_rs::Array, mlx_rs::Array)],
    tokens: i32,
    prev_len: i32,
    scale: f32,
) -> Result<mlx_rs::Array> {
    if blocks.is_empty() {
        bail!("blockwise attention requires at least one cache block");
    }
    let shape = q.shape();
    if shape.len() != 4 {
        bail!("blockwise attention query must be [batch, heads, tokens, dim], got {shape:?}");
    }
    let batch = shape[0];
    let query_heads = shape[1];
    let head_dim = shape[3];
    let first_k_shape = blocks[0].1.shape();
    let first_v_shape = blocks[0].2.shape();
    if first_k_shape.len() != 4 || first_v_shape.len() != 4 {
        bail!("blockwise attention cache must be [batch, heads, tokens, dim]");
    }
    let kv_heads = first_k_shape[1];
    if kv_heads <= 0 || query_heads % kv_heads != 0 {
        bail!("query heads {query_heads} must be divisible by key/value heads {kv_heads}");
    }
    let repeat = query_heads / kv_heads;
    let q_float = q.as_dtype(mlx_rs::Dtype::Float32)?;
    let mut running_max: Option<mlx_rs::Array> = None;
    let mut running_denom: Option<mlx_rs::Array> = None;
    let mut running_acc: Option<mlx_rs::Array> = None;

    for (start, k_block, v_block) in blocks {
        let k_shape = k_block.shape();
        let v_shape = v_block.shape();
        if k_shape.len() != 4 || v_shape.len() != 4 || k_shape != v_shape {
            bail!("blockwise attention cache block shape mismatch: {k_shape:?} vs {v_shape:?}");
        }
        let block_len = k_shape[2];
        let k_block = k_block.as_dtype(mlx_rs::Dtype::Float32)?;
        let v_block = v_block.as_dtype(mlx_rs::Dtype::Float32)?;
        let scores = if repeat > 1 {
            let q_grouped = q_float.reshape(&[batch, kv_heads, repeat, tokens, head_dim])?;
            let k_grouped = k_block.reshape(&[batch, kv_heads, 1, block_len, head_dim])?;
            q_grouped
                .matmul(&k_grouped.transpose_axes(&[0, 1, 2, 4, 3])?)?
                .reshape(&[batch, query_heads, tokens, block_len])?
        } else {
            q_float.matmul(&k_block.transpose_axes(&[0, 1, 3, 2])?)?
        };
        let mut scores = scores.multiply(&mlx_rs::Array::from_f32(scale))?;
        if *start + block_len > prev_len {
            let mask = causal_position_mask(tokens, block_len, prev_len, *start, scores.dtype())?;
            scores = scores.add(&mask)?;
        }
        let local_max = scores.max_axis(-1, Some(true))?;
        let weights = scores.subtract(&local_max)?.exp()?;
        let local_denom = weights.sum_axis(-1, Some(true))?;
        let local_acc = if repeat > 1 {
            weights
                .reshape(&[batch, kv_heads, repeat, tokens, block_len])?
                .matmul(&v_block.reshape(&[batch, kv_heads, 1, block_len, head_dim])?)?
                .reshape(&[batch, query_heads, tokens, head_dim])?
        } else {
            weights.matmul(&v_block)?
        };

        match (running_max.take(), running_denom.take(), running_acc.take()) {
            (Some(prev_max), Some(prev_denom), Some(prev_acc)) => {
                let new_max = mlx_rs::ops::maximum(&prev_max, &local_max)?;
                let old_scale = prev_max.subtract(&new_max)?.exp()?;
                let new_scale = local_max.subtract(&new_max)?.exp()?;
                running_acc = Some(
                    prev_acc
                        .multiply(&old_scale)?
                        .add(&local_acc.multiply(&new_scale)?)?,
                );
                running_denom = Some(
                    prev_denom
                        .multiply(&old_scale)?
                        .add(&local_denom.multiply(&new_scale)?)?,
                );
                running_max = Some(new_max);
            }
            _ => {
                running_max = Some(local_max);
                running_denom = Some(local_denom);
                running_acc = Some(local_acc);
            }
        }
    }

    let acc =
        running_acc.ok_or_else(|| anyhow::anyhow!("blockwise attention produced no output"))?;
    let denom =
        running_denom.ok_or_else(|| anyhow::anyhow!("blockwise attention produced no denom"))?;
    Ok(acc.divide(&denom)?.as_dtype(q.dtype())?)
}

fn causal_position_mask(
    tokens: i32,
    block_len: i32,
    query_start: i32,
    key_start: i32,
    dtype: mlx_rs::Dtype,
) -> Result<mlx_rs::Array> {
    let mut values = Vec::with_capacity((tokens * block_len) as usize);
    for query in 0..tokens {
        let query_position = query_start + query;
        for key in 0..block_len {
            let key_position = key_start + key;
            values.push(if key_position <= query_position {
                0.0_f32
            } else {
                -1.0e9_f32
            });
        }
    }
    Ok(mlx_rs::Array::from_slice(&values, &[1, 1, tokens, block_len]).as_dtype(dtype)?)
}

fn split_full_attention_chunk_tokens(tokens: i32, prev_len: i32) -> Option<i32> {
    let enabled = std::env::var("FERRITE_SPLIT_FULL_ATTN")
        .ok()
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(true);
    if !enabled {
        return None;
    }
    let threshold = std::env::var("FERRITE_SPLIT_FULL_ATTN_THRESHOLD")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|value| *value >= 0)
        .unwrap_or(1024);
    let chunk = std::env::var("FERRITE_SPLIT_FULL_ATTN_CHUNK_TOKENS")
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(128);
    (prev_len >= threshold && tokens > chunk).then_some(chunk)
}

fn split_decode_scaled_dot_product_attention(
    q: &mlx_rs::Array,
    k: &mlx_rs::Array,
    v: &mlx_rs::Array,
    tokens: i32,
    prev_len: i32,
    chunk: i32,
    scale: f32,
) -> Result<mlx_rs::Array> {
    let mut outputs = Vec::new();
    let mut start = 0;
    while start < tokens {
        let end = (start + chunk).min(tokens);
        let key_end = (prev_len + end).min(k.shape()[2]);
        let q_chunk = q.index((.., .., start..end, ..));
        let k_chunk = k.index((.., .., 0..key_end, ..));
        let v_chunk = v.index((.., .., 0..key_end, ..));
        let mask = decode_block_causal_mask(end - start, key_end, prev_len + start, q.dtype())?;
        outputs.push(mlx_rs::fast::scaled_dot_product_attention(
            &q_chunk, &k_chunk, &v_chunk, scale, &mask,
        )?);
        start = end;
    }
    Ok(concatenate_axis(&outputs, 2)?)
}

fn apply_attention_gate(
    output: mlx_rs::Array,
    gate: Option<&mlx_rs::Array>,
) -> Result<mlx_rs::Array> {
    if let Some(gate) = gate {
        Ok(&output * &mlx_rs::nn::sigmoid(gate)?)
    } else {
        Ok(output)
    }
}

#[derive(Debug)]
pub struct LinearAttentionWeights {
    pub in_proj_qkv: crate::mlx_backend::QuantizedLinear,
    pub in_proj_a: crate::mlx_backend::QuantizedLinear,
    pub in_proj_b: crate::mlx_backend::QuantizedLinear,
    pub in_proj_z: crate::mlx_backend::QuantizedLinear,
    pub fused_qkv_z_b_a: Option<crate::mlx_backend::FusedQuantizedLinears>,
    pub out_proj: crate::mlx_backend::QuantizedLinear,
    pub norm: crate::mlx_backend::RmsNorm,
    pub conv1d_weight: mlx_rs::Array,
    pub conv1d_weight_shape: Vec<i32>,
    pub conv1d_weight_f32: Vec<f32>,
    pub a_log: mlx_rs::Array,
    pub a_log_f32: Vec<f32>,
    pub dt_bias: mlx_rs::Array,
    pub dt_bias_f32: Vec<f32>,
    pub metal: Option<Arc<crate::metal_kernels::LinearGdnKernels>>,
}

pub struct LinearAttentionProjection {
    pub q: mlx_rs::Array,
    pub k: mlx_rs::Array,
    pub v: mlx_rs::Array,
    pub a: mlx_rs::Array,
    pub b: mlx_rs::Array,
    pub z: mlx_rs::Array,
}

impl LinearAttentionWeights {
    fn project_qkv_z_b_a(
        &self,
        x: &mlx_rs::Array,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array, mlx_rs::Array, mlx_rs::Array)> {
        if let Some(fused) = &self.fused_qkv_z_b_a {
            let mut parts = fused.forward(x)?;
            if parts.len() != 4 {
                bail!(
                    "fused linear attention projection returned {} parts",
                    parts.len()
                );
            }
            let a = parts.pop().expect("length checked");
            let b = parts.pop().expect("length checked");
            let z = parts.pop().expect("length checked");
            let qkv = parts.pop().expect("length checked");
            return Ok((qkv, z, b, a));
        }

        Ok((
            self.in_proj_qkv.forward(x)?,
            self.in_proj_z.forward(x)?,
            self.in_proj_b.forward(x)?,
            self.in_proj_a.forward(x)?,
        ))
    }

    pub fn project(
        &self,
        x: &mlx_rs::Array,
        linear_num_key_heads: i32,
        linear_key_head_dim: i32,
        linear_num_value_heads: i32,
        linear_value_head_dim: i32,
    ) -> Result<LinearAttentionProjection> {
        let qk_width = linear_num_key_heads * linear_key_head_dim;
        let v_width = linear_num_value_heads * linear_value_head_dim;
        let (qkv, z, b, a) = self.project_qkv_z_b_a(x)?;
        let mut parts = qkv.split_axis(&[qk_width, qk_width * 2], -1)?;
        let q = parts.remove(0);
        let k = parts.remove(0);
        let v = parts.remove(0);
        if v.shape().last().copied() != Some(v_width) {
            bail!(
                "linear attention v width mismatch: got {:?}, expected last dim {v_width}",
                v.shape()
            );
        }
        Ok(LinearAttentionProjection { q, k, v, a, b, z })
    }

    pub fn new_cache(&self, plan: &Qwen36Plan) -> Result<LinearAttentionCache> {
        let (conv_state, recurrent_state, conv_out, recurrent_out, q_normed, k_normed) =
            if self.metal.is_some() {
                (
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                    Vec::new(),
                )
            } else {
                let hk = plan.linear_num_key_heads as usize;
                let hv = plan.linear_num_value_heads as usize;
                let dk = plan.linear_key_head_dim as usize;
                let dv = plan.linear_value_head_dim as usize;
                let conv_dim = hk * dk * 2 + hv * dv;
                let keep = conv_keep(&self.conv1d_weight.shape().to_vec(), conv_dim)?;
                (
                    vec![0.0; keep * conv_dim],
                    vec![0.0; hv * dv * dk],
                    vec![0.0; conv_dim],
                    vec![0.0; hv * dv],
                    vec![0.0; hk * dk],
                    vec![0.0; hk * dk],
                )
            };
        Ok(LinearAttentionCache {
            conv_state,
            recurrent_state,
            conv_out,
            recurrent_out,
            q_normed,
            k_normed,
            metal_conv_state: None,
            metal_recurrent_state: None,
            metal_conv_block_states: None,
            metal_recurrent_block_states: None,
        })
    }

    pub fn forward_reference(&self, x: &mlx_rs::Array, plan: &Qwen36Plan) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("linear attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0] as usize;
        let tokens = shape[1] as usize;
        let hk = plan.linear_num_key_heads as usize;
        let hv = plan.linear_num_value_heads as usize;
        let dk = plan.linear_key_head_dim as usize;
        let dv = plan.linear_value_head_dim as usize;
        if hk == 0 || hv == 0 || dk == 0 || dv == 0 || hv % hk != 0 {
            bail!("invalid linear attention dimensions: hk={hk}, hv={hv}, dk={dk}, dv={dv}");
        }
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;

        let (qkv, z, b, a) = self.project_qkv_z_b_a(x)?;
        if qkv.shape() != [batch as i32, tokens as i32, conv_dim as i32] {
            bail!(
                "linear attention qkv shape mismatch: got {:?}, expected [{batch}, {tokens}, {conv_dim}]",
                qkv.shape()
            );
        }
        if a.shape() != [batch as i32, tokens as i32, hv as i32]
            || b.shape() != [batch as i32, tokens as i32, hv as i32]
        {
            bail!(
                "linear attention a/b shape mismatch: a={:?}, b={:?}, expected [{batch}, {tokens}, {hv}]",
                a.shape(),
                b.shape()
            );
        }

        let qkv = array_to_f32_vec(&qkv)?;
        let a = array_to_f32_vec(&a)?;
        let b = array_to_f32_vec(&b)?;
        if self.a_log_f32.len() != hv || self.dt_bias_f32.len() != hv {
            bail!(
                "linear attention state parameter mismatch: A_log={}, dt_bias={}, expected {hv}",
                self.a_log_f32.len(),
                self.dt_bias_f32.len()
            );
        }

        let keep = conv_keep(&self.conv1d_weight_shape, conv_dim)?;
        let conv_out = depthwise_causal_conv_silu(
            &qkv,
            batch,
            tokens,
            conv_dim,
            keep,
            &self.conv1d_weight_f32,
            &self.conv1d_weight_shape,
        )?;
        let recurrent = gated_delta_reference(
            &conv_out,
            &a,
            &b,
            &self.a_log_f32,
            &self.dt_bias_f32,
            batch,
            tokens,
            hk,
            hv,
            dk,
            dv,
        );
        let out = mlx_rs::Array::from_slice(
            &recurrent,
            &[batch as i32, tokens as i32, hv as i32, dv as i32],
        );
        let z = z.reshape(&[batch as i32, tokens as i32, hv as i32, dv as i32])?;
        let gate = mlx_rs::nn::silu(&z)?;
        let normed = self.norm.forward(&out)?;
        let gated = &gate * &normed;
        self.out_proj
            .forward(&gated.reshape(&[batch as i32, tokens as i32, (hv * dv) as i32])?)
    }

    pub fn forward_reference_with_cache(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        cache: &mut LinearAttentionCache,
    ) -> Result<mlx_rs::Array> {
        if let Some(metal) = &self.metal {
            return self.forward_metal_with_cache(x, plan, cache, metal);
        }

        let shape = x.shape();
        if shape.len() != 3 {
            bail!("linear attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0] as usize;
        if batch != 1 {
            bail!("linear attention cached decode currently supports batch=1, got batch={batch}");
        }
        let tokens = shape[1] as usize;
        let hk = plan.linear_num_key_heads as usize;
        let hv = plan.linear_num_value_heads as usize;
        let dk = plan.linear_key_head_dim as usize;
        let dv = plan.linear_value_head_dim as usize;
        if hk == 0 || hv == 0 || dk == 0 || dv == 0 || hv % hk != 0 {
            bail!("invalid linear attention dimensions: hk={hk}, hv={hv}, dk={dk}, dv={dv}");
        }
        let key_dim = hk * dk;
        let value_dim = hv * dv;
        let conv_dim = key_dim * 2 + value_dim;

        let (qkv, z, b, a) = self.project_qkv_z_b_a(x)?;
        if qkv.shape() != [batch as i32, tokens as i32, conv_dim as i32] {
            bail!(
                "linear attention qkv shape mismatch: got {:?}, expected [{batch}, {tokens}, {conv_dim}]",
                qkv.shape()
            );
        }

        let qkv = array_to_f32_vec(&qkv)?;
        let a = array_to_f32_vec(&a)?;
        let b = array_to_f32_vec(&b)?;
        let keep = conv_keep(&self.conv1d_weight_shape, conv_dim)?;
        let expected_conv_state = keep * conv_dim;
        let expected_recurrent_state = hv * dv * dk;
        if cache.conv_state.len() != expected_conv_state {
            bail!(
                "linear attention conv cache mismatch: got {}, expected {expected_conv_state}",
                cache.conv_state.len()
            );
        }
        if cache.recurrent_state.len() != expected_recurrent_state {
            bail!(
                "linear attention recurrent cache mismatch: got {}, expected {expected_recurrent_state}",
                cache.recurrent_state.len()
            );
        }
        cache.conv_out.resize(tokens * conv_dim, 0.0);
        cache.recurrent_out.resize(tokens * hv * dv, 0.0);
        cache.q_normed.resize(key_dim, 0.0);
        cache.k_normed.resize(key_dim, 0.0);
        depthwise_causal_conv_silu_with_state(
            &qkv,
            tokens,
            conv_dim,
            keep,
            &self.conv1d_weight_f32,
            &self.conv1d_weight_shape,
            &mut cache.conv_state,
            &mut cache.conv_out,
        )?;
        gated_delta_reference_with_state(
            &cache.conv_out,
            &a,
            &b,
            &self.a_log_f32,
            &self.dt_bias_f32,
            tokens,
            hk,
            hv,
            dk,
            dv,
            &mut cache.recurrent_state,
            &mut cache.recurrent_out,
            &mut cache.q_normed,
            &mut cache.k_normed,
        );
        let out = mlx_rs::Array::from_slice(
            &cache.recurrent_out,
            &[batch as i32, tokens as i32, hv as i32, dv as i32],
        );
        let z = z.reshape(&[batch as i32, tokens as i32, hv as i32, dv as i32])?;
        let gate = mlx_rs::nn::silu(&z)?;
        let normed = self.norm.forward(&out)?;
        let gated = &gate * &normed;
        self.out_proj
            .forward(&gated.reshape(&[batch as i32, tokens as i32, (hv * dv) as i32])?)
    }

    fn forward_metal_with_cache(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        cache: &mut LinearAttentionCache,
        metal: &crate::metal_kernels::LinearGdnKernels,
    ) -> Result<mlx_rs::Array> {
        let shape = x.shape();
        if shape.len() != 3 {
            bail!("linear attention input must be [batch, tokens, hidden], got {shape:?}");
        }
        let batch = shape[0];
        if batch != 1 {
            bail!("linear attention cached decode currently supports batch=1, got batch={batch}");
        }
        let tokens = shape[1];
        let hk = plan.linear_num_key_heads as i32;
        let hv = plan.linear_num_value_heads as i32;
        let dk = plan.linear_key_head_dim as i32;
        let dv = plan.linear_value_head_dim as i32;
        if hk == 0 || hv == 0 || dk == 0 || dv == 0 || hv % hk != 0 {
            bail!("invalid linear attention dimensions: hk={hk}, hv={hv}, dk={dk}, dv={dv}");
        }
        let key_dim = hk * dk;
        let conv_dim = key_dim * 2 + hv * dv;

        let (qkv, z, b, a) = self.project_qkv_z_b_a(x)?;
        if qkv.shape() != [batch, tokens, conv_dim] {
            bail!(
                "linear attention qkv shape mismatch: got {:?}, expected [{batch}, {tokens}, {conv_dim}]",
                qkv.shape()
            );
        }

        let keep = conv_keep(&self.conv1d_weight_shape, conv_dim as usize)? as i32;
        let conv_state_shape = [batch, keep, conv_dim];
        if cache
            .metal_conv_state
            .as_ref()
            .is_none_or(|state| state.shape() != conv_state_shape || state.dtype() != qkv.dtype())
        {
            cache.metal_conv_state =
                Some(mlx_rs::ops::zeros_dtype(&conv_state_shape, qkv.dtype())?);
        }
        let recurrent_state_shape = [batch, hv, dv, dk];
        if cache
            .metal_recurrent_state
            .as_ref()
            .is_none_or(|state| state.shape() != recurrent_state_shape)
        {
            cache.metal_recurrent_state = Some(mlx_rs::ops::zeros_dtype(
                &recurrent_state_shape,
                mlx_rs::Dtype::Float32,
            )?);
        }

        let (conv_out, conv_states) = metal.conv1d_silu(
            &qkv,
            cache
                .metal_conv_state
                .as_ref()
                .expect("metal conv state was initialized"),
            &self.conv1d_weight,
            batch,
            tokens,
            conv_dim,
            keep,
        )?;
        cache.metal_conv_state = Some(conv_states.index((.., -1, .., ..)));
        cache.metal_conv_block_states = if tokens > 1 { Some(conv_states) } else { None };

        let (out, recurrent_states) = metal.gated_delta_inline(
            &conv_out,
            &a,
            &b,
            &self.a_log,
            &self.dt_bias,
            cache
                .metal_recurrent_state
                .as_ref()
                .expect("metal recurrent state was initialized"),
            batch,
            tokens,
            hk,
            hv,
            dk,
            dv,
        )?;
        cache.metal_recurrent_state = Some(recurrent_states.index((.., -1, .., .., ..)));
        cache.metal_recurrent_block_states = if tokens > 1 {
            Some(recurrent_states)
        } else {
            None
        };

        let z = z.reshape(&[batch, tokens, hv, dv])?;
        let gate = mlx_rs::nn::silu(&z)?;
        let normed = self.norm.forward(&out)?;
        let gated = &gate * &normed;
        self.out_proj
            .forward(&gated.reshape(&[batch, tokens, hv * dv])?)
    }
}

#[derive(Debug)]
pub struct MlpWeights {
    pub gate_proj: crate::mlx_backend::QuantizedLinear,
    pub up_proj: crate::mlx_backend::QuantizedLinear,
    pub down_proj: crate::mlx_backend::QuantizedLinear,
    pub fused_gate_up: Option<crate::mlx_backend::FusedQuantizedLinears>,
}

impl MlpWeights {
    pub fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        if let Some(out) = crate::mlx_backend::compiled_swiglu_mlp_q4(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        )? {
            return Ok(out);
        }
        if crate::metal_kernels::gate_up_swiglu_qmv4_enabled() {
            if let Some(fused) = crate::metal_kernels::gate_up_swiglu_qmv4_activation(
                x,
                &self.gate_proj,
                &self.up_proj,
            )? {
                return self.down_proj.forward(&fused);
            }
        }
        let (gate_raw, up) = if let Some(fused) = &self.fused_gate_up {
            let mut parts = fused.forward(x)?;
            if parts.len() != 2 {
                bail!("fused MLP projection returned {} parts", parts.len());
            }
            let up = parts.pop().expect("length checked");
            let gate = parts.pop().expect("length checked");
            (gate, up)
        } else {
            (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
        };
        let gate = mlx_rs::nn::silu(&gate_raw)?;
        let fused = &gate * &up;
        self.down_proj.forward(&fused)
    }
}

fn conv_keep(shape: &[i32], conv_dim: usize) -> Result<usize> {
    if shape.len() != 3 || shape[0] != conv_dim as i32 || shape[2] != 1 || shape[1] < 2 {
        bail!("expected conv1d.weight shape [{conv_dim}, keep + 1, 1], got {shape:?}");
    }
    Ok((shape[1] - 1) as usize)
}

fn conv_weight_at(weight: &[f32], shape: &[i32], channel: usize, k: usize) -> f32 {
    let kernel = shape[1] as usize;
    weight[(channel * kernel + k) * shape[2] as usize]
}

fn depthwise_causal_conv_silu(
    qkv: &[f32],
    batch: usize,
    tokens: usize,
    conv_dim: usize,
    keep: usize,
    weight: &[f32],
    weight_shape: &[i32],
) -> Result<Vec<f32>> {
    let expected_weight = conv_dim * (keep + 1);
    if weight.len() != expected_weight {
        bail!(
            "conv1d weight length mismatch: got {}, expected {expected_weight}",
            weight.len()
        );
    }
    let mut state = vec![0.0f32; batch * keep * conv_dim];
    let mut out = vec![0.0f32; batch * tokens * conv_dim];
    for b in 0..batch {
        for t in 0..tokens {
            let qkv_base = (b * tokens + t) * conv_dim;
            let out_base = qkv_base;
            for c in 0..conv_dim {
                let mut acc = 0.0f32;
                for k in 0..keep {
                    acc += state[(b * keep + k) * conv_dim + c]
                        * conv_weight_at(weight, weight_shape, c, k);
                }
                acc += qkv[qkv_base + c] * conv_weight_at(weight, weight_shape, c, keep);
                out[out_base + c] = silu_f32(acc);
            }
            for k in 0..keep {
                for c in 0..conv_dim {
                    let value = if k + 1 < keep {
                        state[(b * keep + k + 1) * conv_dim + c]
                    } else {
                        qkv[qkv_base + c]
                    };
                    state[(b * keep + k) * conv_dim + c] = value;
                }
            }
        }
    }
    Ok(out)
}

fn depthwise_causal_conv_silu_with_state(
    qkv: &[f32],
    tokens: usize,
    conv_dim: usize,
    keep: usize,
    weight: &[f32],
    weight_shape: &[i32],
    state: &mut [f32],
    out: &mut [f32],
) -> Result<()> {
    let expected_weight = conv_dim * (keep + 1);
    if weight.len() != expected_weight {
        bail!(
            "conv1d weight length mismatch: got {}, expected {expected_weight}",
            weight.len()
        );
    }
    if state.len() != keep * conv_dim {
        bail!(
            "conv state length mismatch: got {}, expected {}",
            state.len(),
            keep * conv_dim
        );
    }
    if out.len() != tokens * conv_dim {
        bail!(
            "conv output length mismatch: got {}, expected {}",
            out.len(),
            tokens * conv_dim
        );
    }
    for t in 0..tokens {
        let qkv_base = t * conv_dim;
        let out_base = qkv_base;
        for c in 0..conv_dim {
            let mut acc = 0.0f32;
            for k in 0..keep {
                acc += state[k * conv_dim + c] * conv_weight_at(weight, weight_shape, c, k);
            }
            acc += qkv[qkv_base + c] * conv_weight_at(weight, weight_shape, c, keep);
            out[out_base + c] = silu_f32(acc);
        }
        for k in 0..keep {
            for c in 0..conv_dim {
                let value = if k + 1 < keep {
                    state[(k + 1) * conv_dim + c]
                } else {
                    qkv[qkv_base + c]
                };
                state[k * conv_dim + c] = value;
            }
        }
    }
    Ok(())
}

fn gated_delta_reference(
    conv_out: &[f32],
    a: &[f32],
    b_gate: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    batch: usize,
    tokens: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
) -> Vec<f32> {
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let value_heads_per_key = hv / hk;
    let inv_sqrt_dk = 1.0f32 / (dk as f32).sqrt();
    let q_scale = inv_sqrt_dk * inv_sqrt_dk;
    let k_scale = inv_sqrt_dk;
    let mut state = vec![0.0f32; batch * hv * dv * dk];
    let mut out = vec![0.0f32; batch * tokens * hv * dv];
    let mut q_normed = vec![0.0f32; key_dim];
    let mut k_normed = vec![0.0f32; key_dim];

    for batch_index in 0..batch {
        for token_index in 0..tokens {
            let conv_base = (batch_index * tokens + token_index) * conv_dim;
            for key_head in 0..hk {
                let mut q_sum = 0.0f32;
                let mut k_sum = 0.0f32;
                for dim in 0..dk {
                    let q = conv_out[conv_base + key_head * dk + dim];
                    let k = conv_out[conv_base + key_dim + key_head * dk + dim];
                    q_sum += q * q;
                    k_sum += k * k;
                }
                let q_inv = (q_sum / dk as f32 + 1e-6).sqrt().recip();
                let k_inv = (k_sum / dk as f32 + 1e-6).sqrt().recip();
                for dim in 0..dk {
                    let q = conv_out[conv_base + key_head * dk + dim];
                    let k = conv_out[conv_base + key_dim + key_head * dk + dim];
                    q_normed[key_head * dk + dim] = q * q_inv * q_scale;
                    k_normed[key_head * dk + dim] = k * k_inv * k_scale;
                }
            }

            for value_head in 0..hv {
                let key_head = value_head / value_heads_per_key;
                let a_index = (batch_index * tokens + token_index) * hv + value_head;
                let beta = sigmoid_f32(b_gate[a_index]);
                let g = (-a_log[value_head].exp() * softplus_f32(a[a_index] + dt_bias[value_head]))
                    .exp();
                for value_dim_index in 0..dv {
                    let value =
                        conv_out[conv_base + 2 * key_dim + value_head * dv + value_dim_index];
                    let state_base = ((batch_index * hv + value_head) * dv + value_dim_index) * dk;
                    let key_base = key_head * dk;
                    let mut kv_mem = 0.0f32;
                    for dim in 0..dk {
                        state[state_base + dim] *= g;
                        kv_mem += state[state_base + dim] * k_normed[key_base + dim];
                    }
                    let delta = (value - kv_mem) * beta;
                    let mut y = 0.0f32;
                    for dim in 0..dk {
                        state[state_base + dim] += k_normed[key_base + dim] * delta;
                        y += state[state_base + dim] * q_normed[key_base + dim];
                    }
                    out[((batch_index * tokens + token_index) * hv + value_head) * dv
                        + value_dim_index] = y;
                }
            }
        }
    }

    out
}

fn gated_delta_reference_with_state(
    conv_out: &[f32],
    a: &[f32],
    b_gate: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    tokens: usize,
    hk: usize,
    hv: usize,
    dk: usize,
    dv: usize,
    state: &mut [f32],
    out: &mut [f32],
    q_normed: &mut [f32],
    k_normed: &mut [f32],
) {
    let key_dim = hk * dk;
    let value_dim = hv * dv;
    let conv_dim = key_dim * 2 + value_dim;
    let value_heads_per_key = hv / hk;
    let inv_sqrt_dk = 1.0f32 / (dk as f32).sqrt();
    let q_scale = inv_sqrt_dk * inv_sqrt_dk;
    let k_scale = inv_sqrt_dk;

    for token_index in 0..tokens {
        let conv_base = token_index * conv_dim;
        for key_head in 0..hk {
            let mut q_sum = 0.0f32;
            let mut k_sum = 0.0f32;
            for dim in 0..dk {
                let q = conv_out[conv_base + key_head * dk + dim];
                let k = conv_out[conv_base + key_dim + key_head * dk + dim];
                q_sum += q * q;
                k_sum += k * k;
            }
            let q_inv = (q_sum / dk as f32 + 1e-6).sqrt().recip();
            let k_inv = (k_sum / dk as f32 + 1e-6).sqrt().recip();
            for dim in 0..dk {
                let q = conv_out[conv_base + key_head * dk + dim];
                let k = conv_out[conv_base + key_dim + key_head * dk + dim];
                q_normed[key_head * dk + dim] = q * q_inv * q_scale;
                k_normed[key_head * dk + dim] = k * k_inv * k_scale;
            }
        }

        let value_base = conv_base + 2 * key_dim;
        let out_base = token_index * hv * dv;
        for value_head in 0..hv {
            let key_head = value_head / value_heads_per_key;
            let a_index = token_index * hv + value_head;
            let beta = sigmoid_f32(b_gate[a_index]);
            let g =
                (-a_log[value_head].exp() * softplus_f32(a[a_index] + dt_bias[value_head])).exp();
            let key_base = key_head * dk;
            for value_dim_index in 0..dv {
                let value = conv_out[value_base + value_head * dv + value_dim_index];
                let state_base = (value_head * dv + value_dim_index) * dk;
                let mut kv_mem = 0.0f32;
                for dim in 0..dk {
                    state[state_base + dim] *= g;
                    kv_mem += state[state_base + dim] * k_normed[key_base + dim];
                }
                let delta = (value - kv_mem) * beta;
                let mut y = 0.0f32;
                for dim in 0..dk {
                    state[state_base + dim] += k_normed[key_base + dim] * delta;
                    y += state[state_base + dim] * q_normed[key_base + dim];
                }
                out[out_base + value_head * dv + value_dim_index] = y;
            }
        }
    }
}

fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}
