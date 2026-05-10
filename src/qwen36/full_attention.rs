use anyhow::{Result, bail};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;

use super::FullAttentionCache;
use super::masks::decode_block_causal_mask;

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
