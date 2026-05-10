#![allow(dead_code)]

use std::collections::BTreeSet;
#[cfg(feature = "native-mlx")]
use std::sync::Arc;
#[cfg(feature = "native-mlx")]
use std::time::Instant;

use anyhow::{Result, bail};
#[cfg(feature = "native-mlx")]
use mlx_rs::ops::concatenate_axis;
#[cfg(feature = "native-mlx")]
use mlx_rs::ops::indexing::IndexOp;
use serde::Serialize;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum LayerKind {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, Serialize)]
pub struct LayerPlan {
    pub index: u32,
    pub kind: LayerKind,
}

#[derive(Clone, Debug, Serialize)]
pub struct Qwen36Plan {
    pub num_layers: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub vocab_size: u32,
    pub head_dim: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub rope_dimensions: u32,
    pub rope_theta: f32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub full_attention_layers: Vec<u32>,
    pub linear_attention_layers: Vec<u32>,
    pub layers: Vec<LayerPlan>,
}

impl Qwen36Plan {
    pub fn from_model(model: &crate::model::LoadedModel) -> Result<Self> {
        let cfg = model.config.text();
        if cfg.model_type.as_deref() != Some("qwen3_5_text") {
            bail!(
                "unsupported model_type for Qwen3.6 plan: {:?}",
                cfg.model_type
            );
        }
        let num_layers = required(cfg.num_hidden_layers, "num_hidden_layers")?;
        let hidden_size = required(cfg.hidden_size, "hidden_size")?;
        let intermediate_size = required(cfg.intermediate_size, "intermediate_size")?;
        let vocab_size = required(cfg.vocab_size, "vocab_size")?;
        let head_dim = required(cfg.head_dim, "head_dim")?;
        let num_attention_heads = required(cfg.num_attention_heads, "num_attention_heads")?;
        let num_key_value_heads = required(cfg.num_key_value_heads, "num_key_value_heads")?;
        let partial_rotary_factor = cfg.partial_rotary_factor.unwrap_or(1.0);
        let rope_dimensions = ((head_dim as f32) * partial_rotary_factor).round() as u32;
        let rope_theta = cfg
            .rope_parameters
            .get("rope_theta")
            .and_then(|value| value.as_f64())
            .unwrap_or(1_000_000.0) as f32;
        let linear_key_head_dim = required(cfg.linear_key_head_dim, "linear_key_head_dim")?;
        let linear_value_head_dim = required(cfg.linear_value_head_dim, "linear_value_head_dim")?;
        let linear_num_key_heads = required(cfg.linear_num_key_heads, "linear_num_key_heads")?;
        let linear_num_value_heads =
            required(cfg.linear_num_value_heads, "linear_num_value_heads")?;

        let keys = model
            .tensors
            .model_tensors
            .keys()
            .map(String::as_str)
            .collect::<BTreeSet<_>>();

        let mut full_attention_layers = Vec::new();
        let mut linear_attention_layers = Vec::new();
        let mut layers = Vec::new();
        for index in 0..num_layers {
            let prefix = format!("language_model.model.layers.{index}.");
            let has_linear = keys
                .iter()
                .any(|key| key.starts_with(&(prefix.clone() + "linear_attn.")));
            let has_full = keys
                .iter()
                .any(|key| key.starts_with(&(prefix.clone() + "self_attn.")));
            let kind = match (has_linear, has_full) {
                (true, false) => LayerKind::LinearAttention,
                (false, true) => LayerKind::FullAttention,
                (true, true) => bail!("layer {index} has both linear_attn and self_attn tensors"),
                (false, false) => bail!("layer {index} has no attention tensors"),
            };
            match kind {
                LayerKind::LinearAttention => linear_attention_layers.push(index),
                LayerKind::FullAttention => full_attention_layers.push(index),
            }
            layers.push(LayerPlan { index, kind });
        }

        Ok(Self {
            num_layers,
            hidden_size,
            intermediate_size,
            vocab_size,
            head_dim,
            num_attention_heads,
            num_key_value_heads,
            rope_dimensions,
            rope_theta,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_num_key_heads,
            linear_num_value_heads,
            full_attention_layers,
            linear_attention_layers,
            layers,
        })
    }
}

fn required(value: Option<u32>, name: &str) -> Result<u32> {
    value.ok_or_else(|| anyhow::anyhow!("missing {name} in model config"))
}

#[cfg(feature = "native-mlx")]
fn env_enabled(name: &str) -> bool {
    env_flag(name, false)
}

#[cfg(feature = "native-mlx")]
fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
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
#[derive(Debug)]
pub struct Qwen36Weights {
    pub embeddings: crate::mlx_backend::QuantizedEmbedding,
    pub final_norm: crate::mlx_backend::RmsNorm,
    pub lm_head: crate::mlx_backend::QuantizedLinear,
    pub draft_lm_head: Option<crate::mlx_backend::QuantizedLinear>,
    pub layers: Vec<LayerWeights>,
    pub mtp: Option<MtpWeights>,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug)]
pub struct DecodeState {
    pub position: i32,
    pub layers: Vec<LayerDecodeState>,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug)]
pub enum LayerDecodeState {
    Full(FullAttentionCache),
    Linear(LinearAttentionCache),
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug, Default)]
pub struct FullAttentionCache {
    pub k: Option<mlx_rs::Array>,
    pub v: Option<mlx_rs::Array>,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug)]
pub struct LinearAttentionCache {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub recurrent_out: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub metal_conv_state: Option<mlx_rs::Array>,
    pub metal_recurrent_state: Option<mlx_rs::Array>,
    pub metal_conv_block_states: Option<mlx_rs::Array>,
    pub metal_recurrent_block_states: Option<mlx_rs::Array>,
}

#[cfg(feature = "native-mlx")]
impl DecodeState {
    pub fn clear_transient_block_states(&mut self) {
        for layer in &mut self.layers {
            if let LayerDecodeState::Linear(cache) = layer {
                cache.metal_conv_block_states = None;
                cache.metal_recurrent_block_states = None;
            }
        }
    }

    pub fn truncate_after_decode_block(
        &mut self,
        base_position: i32,
        keep_tokens: i32,
    ) -> Result<()> {
        if keep_tokens < 0 {
            bail!("cannot keep a negative number of decode tokens: {keep_tokens}");
        }
        let decoded_tokens = self.position - base_position;
        if decoded_tokens < keep_tokens {
            bail!(
                "cannot truncate decode state to {keep_tokens} tokens after base position {base_position}; state only decoded {decoded_tokens}"
            );
        }
        let new_position = base_position + keep_tokens;
        for layer in &mut self.layers {
            match layer {
                LayerDecodeState::Full(cache) => {
                    if let Some(k) = cache.k.take() {
                        let current = k.shape().get(2).copied().unwrap_or(0);
                        cache.k = Some(if current > new_position {
                            k.index((.., .., 0..new_position, ..))
                        } else {
                            k
                        });
                    }
                    if let Some(v) = cache.v.take() {
                        let current = v.shape().get(2).copied().unwrap_or(0);
                        cache.v = Some(if current > new_position {
                            v.index((.., .., 0..new_position, ..))
                        } else {
                            v
                        });
                    }
                }
                LayerDecodeState::Linear(cache) => {
                    if keep_tokens == decoded_tokens {
                        cache.metal_conv_block_states = None;
                        cache.metal_recurrent_block_states = None;
                        continue;
                    }
                    if keep_tokens == 0 {
                        bail!("linear attention decode state cannot truncate back to block start");
                    }
                    let row = keep_tokens - 1;
                    let conv_states = cache
                        .metal_conv_block_states
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing Metal conv block states"))?;
                    let recurrent_states = cache
                        .metal_recurrent_block_states
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing Metal recurrent block states"))?;
                    cache.metal_conv_state = Some(conv_states.index((.., row, .., ..)));
                    cache.metal_recurrent_state =
                        Some(recurrent_states.index((.., row, .., .., ..)));
                }
            }
        }
        self.position = new_position;
        Ok(())
    }
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
pub struct MtpWeights {
    pub pre_fc_norm_embedding: crate::mlx_backend::RmsNorm,
    pub pre_fc_norm_hidden: crate::mlx_backend::RmsNorm,
    pub fc: crate::mlx_backend::Linear,
    pub layer: LayerWeights,
    pub norm: crate::mlx_backend::RmsNorm,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug, Default)]
pub struct MtpDecodeState {
    pub position: i32,
    pub cache: FullAttentionCache,
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug, Default)]
pub struct DecodeProfileTimings {
    pub blocks: u32,
    pub tokens: u32,
    pub embedding_s: f64,
    pub full_attention_s: f64,
    pub linear_attention_s: f64,
    pub mlp_s: f64,
    pub layer_glue_s: f64,
    pub final_norm_s: f64,
    pub lm_head_s: f64,
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
pub struct LayerWeights {
    pub input_norm: crate::mlx_backend::RmsNorm,
    pub post_attention_norm: crate::mlx_backend::RmsNorm,
    pub attention: AttentionWeights,
    pub mlp: MlpWeights,
}

#[cfg(feature = "native-mlx")]
impl LayerWeights {
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
        let hidden = x + &attn_out;
        let mlp_normed = self.post_attention_norm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_normed)?;
        Ok(&hidden + &mlp_out)
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
        let hidden = x + &attn_out;
        let mlp_normed = self.post_attention_norm.forward(&hidden)?;
        let mlp_out = self.mlp.forward(&mlp_normed)?;
        Ok(&hidden + &mlp_out)
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
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
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
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
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
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
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
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
            }
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let normed = self.input_norm.forward(x)?;
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                let hidden = x + &attn_out;
                let mlp_normed = self.post_attention_norm.forward(&hidden)?;
                let mlp_out = self.mlp.forward(&mlp_normed)?;
                Ok(&hidden + &mlp_out)
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

                let started = Instant::now();
                let hidden = x + &attn_out;
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
            (AttentionWeights::Linear(attn), LayerDecodeState::Linear(cache)) => {
                let started = Instant::now();
                let normed = self.input_norm.forward(x)?;
                normed.eval()?;
                profile.layer_glue_s += started.elapsed().as_secs_f64();

                let started = Instant::now();
                let attn_out = attn.forward_reference_with_cache(&normed, plan, cache)?;
                attn_out.eval()?;
                profile.linear_attention_s += started.elapsed().as_secs_f64();

                let started = Instant::now();
                let hidden = x + &attn_out;
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
            _ => bail!("decode state kind does not match layer kind"),
        }
    }
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
pub enum AttentionWeights {
    Full(FullAttentionWeights),
    Linear(LinearAttentionWeights),
}

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
pub struct FullAttentionWeights {
    pub q_proj: crate::mlx_backend::QuantizedLinear,
    pub k_proj: crate::mlx_backend::QuantizedLinear,
    pub v_proj: crate::mlx_backend::QuantizedLinear,
    pub o_proj: crate::mlx_backend::QuantizedLinear,
    pub q_norm: crate::mlx_backend::RmsNorm,
    pub k_norm: crate::mlx_backend::RmsNorm,
}

#[cfg(feature = "native-mlx")]
pub struct FullAttentionProjection {
    pub q: mlx_rs::Array,
    pub k: mlx_rs::Array,
    pub v: mlx_rs::Array,
    pub gate: Option<mlx_rs::Array>,
}

#[cfg(feature = "native-mlx")]
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
        cache.k = Some(k.clone());
        cache.v = Some(projected.v.clone());
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
        let next_k = match cache.k.take() {
            Some(prev) => concatenate_axis(&[prev, k], 2)?,
            None => k,
        };
        let next_v = match cache.v.take() {
            Some(prev) => concatenate_axis(&[prev, projected.v], 2)?,
            None => projected.v,
        };
        cache.k = Some(next_k);
        cache.v = Some(next_v);
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            cache.k.as_ref().expect("k cache was just set"),
            cache.v.as_ref().expect("v cache was just set"),
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
        let prev_k = cache.k.take();
        let prev_len = prev_k.as_ref().map(|prev| prev.shape()[2]).unwrap_or(0);
        let next_k = match prev_k {
            Some(prev) => concatenate_axis(&[prev, k], 2)?,
            None => k,
        };
        let next_v = match cache.v.take() {
            Some(prev) => concatenate_axis(&[prev, projected.v], 2)?,
            None => projected.v,
        };
        let total_keys = next_k.shape()[2];
        cache.k = Some(next_k);
        cache.v = Some(next_v);
        let mask = decode_block_causal_mask(tokens, total_keys, prev_len, q.dtype())?;
        let attended = mlx_rs::fast::scaled_dot_product_attention(
            &q,
            cache.k.as_ref().expect("k cache was just set"),
            cache.v.as_ref().expect("v cache was just set"),
            1.0 / (head_dim as f32).sqrt(),
            &mask,
        )?;
        let merged = attended.transpose_axes(&[0, 2, 1, 3])?.reshape(&[
            batch,
            tokens,
            num_heads * head_dim,
        ])?;
        let merged = apply_attention_gate(merged, projected.gate.as_ref())?;
        self.o_proj.forward(&merged)
    }
}

#[cfg(feature = "native-mlx")]
fn decode_block_causal_mask(
    tokens: i32,
    total_keys: i32,
    prev_len: i32,
    dtype: mlx_rs::Dtype,
) -> Result<mlx_rs::Array> {
    let mut values = Vec::with_capacity((tokens * total_keys) as usize);
    for query in 0..tokens {
        let max_key = prev_len + query;
        for key in 0..total_keys {
            values.push(if key <= max_key { 0.0_f32 } else { -1.0e9_f32 });
        }
    }
    Ok(mlx_rs::Array::from_slice(&values, &[1, 1, tokens, total_keys]).as_dtype(dtype)?)
}

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
pub struct LinearAttentionProjection {
    pub q: mlx_rs::Array,
    pub k: mlx_rs::Array,
    pub v: mlx_rs::Array,
    pub a: mlx_rs::Array,
    pub b: mlx_rs::Array,
    pub z: mlx_rs::Array,
}

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
#[derive(Debug)]
pub struct MlpWeights {
    pub gate_proj: crate::mlx_backend::QuantizedLinear,
    pub up_proj: crate::mlx_backend::QuantizedLinear,
    pub down_proj: crate::mlx_backend::QuantizedLinear,
    pub fused_gate_up: Option<crate::mlx_backend::FusedQuantizedLinears>,
}

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
impl MtpWeights {
    fn from_store(
        store: &crate::mlx_backend::MlxWeightStore,
        eps: f32,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let prefix = "mtp";
        let layer_prefix = "mtp.layers.0";
        let gate_proj =
            store.quantized_linear(format!("{layer_prefix}.mlp.gate_proj"), group_size, bits)?;
        let up_proj =
            store.quantized_linear(format!("{layer_prefix}.mlp.up_proj"), group_size, bits)?;
        let fused_gate_up = if env_enabled("MTPLX_FUSE_MLP_PROJECTIONS") {
            crate::mlx_backend::FusedQuantizedLinears::try_new(&[&gate_proj, &up_proj])?
        } else {
            None
        };
        Ok(Self {
            pre_fc_norm_embedding: mtp_norm(
                store,
                &format!("{prefix}.pre_fc_norm_embedding"),
                eps,
            )?,
            pre_fc_norm_hidden: mtp_norm(store, &format!("{prefix}.pre_fc_norm_hidden"), eps)?,
            fc: crate::mlx_backend::Linear::from_store(store, format!("{prefix}.fc"))?,
            layer: LayerWeights {
                input_norm: mtp_norm(store, &format!("{layer_prefix}.input_layernorm"), eps)?,
                post_attention_norm: mtp_norm(
                    store,
                    &format!("{layer_prefix}.post_attention_layernorm"),
                    eps,
                )?,
                attention: AttentionWeights::Full(FullAttentionWeights {
                    q_proj: store.quantized_linear(
                        format!("{layer_prefix}.self_attn.q_proj"),
                        group_size,
                        bits,
                    )?,
                    k_proj: store.quantized_linear(
                        format!("{layer_prefix}.self_attn.k_proj"),
                        group_size,
                        bits,
                    )?,
                    v_proj: store.quantized_linear(
                        format!("{layer_prefix}.self_attn.v_proj"),
                        group_size,
                        bits,
                    )?,
                    o_proj: store.quantized_linear(
                        format!("{layer_prefix}.self_attn.o_proj"),
                        group_size,
                        bits,
                    )?,
                    q_norm: mtp_norm(store, &format!("{layer_prefix}.self_attn.q_norm"), eps)?,
                    k_norm: mtp_norm(store, &format!("{layer_prefix}.self_attn.k_norm"), eps)?,
                }),
                mlp: MlpWeights {
                    gate_proj,
                    up_proj,
                    down_proj: store.quantized_linear(
                        format!("{layer_prefix}.mlp.down_proj"),
                        group_size,
                        bits,
                    )?,
                    fused_gate_up,
                },
            },
            norm: mtp_norm(store, &format!("{prefix}.norm"), eps)?,
        })
    }

    fn forward_decode_step(
        &self,
        token_embedding: &mlx_rs::Array,
        target_hidden: &mlx_rs::Array,
        plan: &Qwen36Plan,
        state: &mut MtpDecodeState,
    ) -> Result<mlx_rs::Array> {
        let e = self.pre_fc_norm_embedding.forward(token_embedding)?;
        let h = self.pre_fc_norm_hidden.forward(target_hidden)?;
        let joined = concatenate_axis(&[e, h], -1)?;
        let x = self.fc.forward(&joined)?;
        let AttentionWeights::Full(attn) = &self.layer.attention else {
            bail!("MTP layer must be full attention");
        };
        let normed = self.layer.input_norm.forward(&x)?;
        let attn_out = attn.forward_decode_step(
            &normed,
            plan.num_attention_heads as i32,
            plan.num_key_value_heads as i32,
            plan.head_dim as i32,
            plan.rope_dimensions as i32,
            plan.rope_theta,
            state.position,
            &mut state.cache,
        )?;
        state.position += 1;
        let hidden = &x + &attn_out;
        let mlp_normed = self.layer.post_attention_norm.forward(&hidden)?;
        let mlp_out = self.layer.mlp.forward(&mlp_normed)?;
        let hidden = &hidden + &mlp_out;
        self.norm.forward(&hidden)
    }
}

#[cfg(feature = "native-mlx")]
impl Qwen36Weights {
    pub fn from_loaded(
        model: &crate::model::LoadedModel,
        store: &crate::mlx_backend::MlxWeightStore,
    ) -> Result<Self> {
        let plan = Qwen36Plan::from_model(model)?;
        let eps = model.config.text().rms_norm_eps.unwrap_or(1e-6);
        let group_size = quant_i32(&model.config.quantization, "group_size", 64);
        let bits = quant_i32(&model.config.quantization, "bits", 4);
        let mtp_group_size = quant_i32(
            &model.config.mtplx_mtp_quantization,
            "group_size",
            group_size,
        );
        let mtp_bits = quant_i32(&model.config.mtplx_mtp_quantization, "bits", bits);
        let root = "language_model";
        let quant_spec_for =
            |prefix: &str| quant_spec(&model.config.quantization, prefix, group_size, bits);
        let qlinear = |prefix: &str| {
            let spec = quant_spec_for(prefix);
            store.quantized_linear(prefix, spec.group_size, spec.bits)
        };
        let embeddings_prefix = format!("{root}.model.embed_tokens");
        let embeddings_spec = quant_spec_for(&embeddings_prefix);
        let embeddings = crate::mlx_backend::QuantizedEmbedding::from_store(
            store,
            embeddings_prefix,
            embeddings_spec.group_size,
            embeddings_spec.bits,
        )?;
        let final_norm = norm(store, &format!("{root}.model.norm"), eps)?;
        let lm_head = qlinear(&format!("{root}.lm_head"))?;
        let draft_lm_head = draft_lm_head_from_env(model, &lm_head)?;
        let linear_gdn_metal = if crate::metal_kernels::metal_is_available() {
            Some(Arc::new(crate::metal_kernels::LinearGdnKernels::new()?))
        } else {
            None
        };
        let mut layers = Vec::with_capacity(plan.layers.len());
        for layer in &plan.layers {
            let prefix = format!("{root}.model.layers.{}", layer.index);
            let input_norm = norm(store, &format!("{prefix}.input_layernorm"), eps)?;
            let post_attention_norm =
                norm(store, &format!("{prefix}.post_attention_layernorm"), eps)?;
            let gate_proj = qlinear(&format!("{prefix}.mlp.gate_proj"))?;
            let up_proj = qlinear(&format!("{prefix}.mlp.up_proj"))?;
            let fused_gate_up = if env_enabled("MTPLX_FUSE_MLP_PROJECTIONS") {
                crate::mlx_backend::FusedQuantizedLinears::try_new(&[&gate_proj, &up_proj])?
            } else {
                None
            };
            let mlp = MlpWeights {
                gate_proj,
                up_proj,
                down_proj: qlinear(&format!("{prefix}.mlp.down_proj"))?,
                fused_gate_up,
            };
            let attention = match layer.kind {
                LayerKind::FullAttention => AttentionWeights::Full(FullAttentionWeights {
                    q_proj: qlinear(&format!("{prefix}.self_attn.q_proj"))?,
                    k_proj: qlinear(&format!("{prefix}.self_attn.k_proj"))?,
                    v_proj: qlinear(&format!("{prefix}.self_attn.v_proj"))?,
                    o_proj: qlinear(&format!("{prefix}.self_attn.o_proj"))?,
                    q_norm: norm(store, &format!("{prefix}.self_attn.q_norm"), eps)?,
                    k_norm: norm(store, &format!("{prefix}.self_attn.k_norm"), eps)?,
                }),
                LayerKind::LinearAttention => {
                    let conv1d_weight = store
                        .array(&format!("{prefix}.linear_attn.conv1d.weight"))?
                        .clone();
                    let conv1d_weight_shape = conv1d_weight.shape().to_vec();
                    let conv1d_weight_f32 = array_to_f32_vec(&conv1d_weight)?;
                    let a_log = store.array(&format!("{prefix}.linear_attn.A_log"))?.clone();
                    let a_log_f32 = array_to_f32_vec(&a_log)?;
                    let dt_bias = store
                        .array(&format!("{prefix}.linear_attn.dt_bias"))?
                        .clone();
                    let dt_bias_f32 = array_to_f32_vec(&dt_bias)?;
                    let in_proj_qkv = qlinear(&format!("{prefix}.linear_attn.in_proj_qkv"))?;
                    let in_proj_a = qlinear(&format!("{prefix}.linear_attn.in_proj_a"))?;
                    let in_proj_b = qlinear(&format!("{prefix}.linear_attn.in_proj_b"))?;
                    let in_proj_z = qlinear(&format!("{prefix}.linear_attn.in_proj_z"))?;
                    let fused_qkv_z_b_a = if env_enabled("MTPLX_FUSE_GDN_PROJECTIONS") {
                        crate::mlx_backend::FusedQuantizedLinears::try_new(&[
                            &in_proj_qkv,
                            &in_proj_z,
                            &in_proj_b,
                            &in_proj_a,
                        ])?
                    } else {
                        None
                    };
                    AttentionWeights::Linear(LinearAttentionWeights {
                        in_proj_qkv,
                        in_proj_a,
                        in_proj_b,
                        in_proj_z,
                        fused_qkv_z_b_a,
                        out_proj: qlinear(&format!("{prefix}.linear_attn.out_proj"))?,
                        norm: norm(store, &format!("{prefix}.linear_attn.norm"), eps)?,
                        conv1d_weight,
                        conv1d_weight_shape,
                        conv1d_weight_f32,
                        a_log,
                        a_log_f32,
                        dt_bias,
                        dt_bias_f32,
                        metal: linear_gdn_metal.clone(),
                    })
                }
            };
            layers.push(LayerWeights {
                input_norm,
                post_attention_norm,
                attention,
                mlp,
            });
        }
        let mtp = if model.tensors.mtp_tensors.contains_key("mtp.fc.weight") {
            Some(MtpWeights::from_store(
                store,
                eps,
                mtp_group_size,
                mtp_bits,
            )?)
        } else {
            None
        };
        Ok(Self {
            embeddings,
            final_norm,
            lm_head,
            draft_lm_head,
            layers,
            mtp,
        })
    }

    pub fn forward_ablate_linear_attention(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
    ) -> Result<mlx_rs::Array> {
        let mut hidden = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward_with_linear_attention_ablation(&hidden, plan, 0)?;
        }
        let post_norm = self.final_norm.forward(&hidden)?;
        self.lm_head.forward(&post_norm)
    }

    pub fn forward_reference(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
    ) -> Result<mlx_rs::Array> {
        let hidden = self.forward_reference_hidden(input_ids, plan)?;
        let post_norm = self.final_norm.forward(&hidden)?;
        self.lm_head.forward(&post_norm)
    }

    pub fn forward_reference_hidden(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
    ) -> Result<mlx_rs::Array> {
        let mut hidden = self.embeddings.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward_reference(&hidden, plan, 0)?;
        }
        Ok(hidden)
    }

    pub fn forward_reference_last_logits(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
    ) -> Result<mlx_rs::Array> {
        let hidden = self.forward_reference_hidden(input_ids, plan)?;
        let last_hidden = hidden.index((.., -1.., ..));
        let post_norm = self.final_norm.forward(&last_hidden)?;
        self.lm_head.forward(&post_norm)
    }

    pub fn new_decode_state(&self, plan: &Qwen36Plan) -> Result<DecodeState> {
        let mut layers = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match &layer.attention {
                AttentionWeights::Full(_) => {
                    layers.push(LayerDecodeState::Full(FullAttentionCache::default()))
                }
                AttentionWeights::Linear(attn) => {
                    let (conv_state, recurrent_state, conv_out, recurrent_out, q_normed, k_normed) =
                        if attn.metal.is_some() {
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
                            let keep = conv_keep(&attn.conv1d_weight.shape().to_vec(), conv_dim)?;
                            (
                                vec![0.0; keep * conv_dim],
                                vec![0.0; hv * dv * dk],
                                vec![0.0; conv_dim],
                                vec![0.0; hv * dv],
                                vec![0.0; hk * dk],
                                vec![0.0; hk * dk],
                            )
                        };
                    layers.push(LayerDecodeState::Linear(LinearAttentionCache {
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
                    }));
                }
            }
        }
        Ok(DecodeState {
            position: 0,
            layers,
        })
    }

    pub fn prefill_decode_state(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<mlx_rs::Array> {
        self.prefill_decode_state_with_hidden(input_ids, plan, state)
            .map(|(logits, _)| logits)
    }

    pub fn prefill_decode_state_with_hidden(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        self.prefill_decode_state_with_hidden_sequence(input_ids, plan, state)
            .map(|(logits, last_hidden, _)| (logits, last_hidden))
    }

    pub fn prefill_decode_state_with_hidden_sequence(
        &self,
        input_ids: &mlx_rs::Array,
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array, mlx_rs::Array)> {
        let shape = input_ids.shape();
        if shape.len() != 2 || shape[0] != 1 {
            bail!("prefill input ids must be [1, tokens], got {shape:?}");
        }
        if state.layers.len() != self.layers.len() {
            bail!("decode state layer count mismatch");
        }
        let mut hidden = self.embeddings.forward(input_ids)?;
        for (layer, cache) in self.layers.iter().zip(state.layers.iter_mut()) {
            hidden = layer.forward_prefill(&hidden, plan, 0, cache)?;
        }
        state.position = shape[1];
        let post_norm = self.final_norm.forward(&hidden)?;
        let last_hidden = post_norm.index((.., -1.., ..));
        let logits = self.lm_head.forward(&last_hidden)?;
        Ok((logits, last_hidden, post_norm))
    }

    pub fn decode_step_logits(
        &self,
        input_id: u32,
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<mlx_rs::Array> {
        self.decode_step_logits_with_hidden(input_id, plan, state)
            .map(|(logits, _)| logits)
    }

    pub fn decode_step_logits_with_hidden(
        &self,
        input_id: u32,
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        if state.layers.len() != self.layers.len() {
            bail!("decode state layer count mismatch");
        }
        let input = [input_id as i32];
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, 1]);
        let mut hidden = self.embeddings.forward(&input_ids)?;
        for (layer, cache) in self.layers.iter().zip(state.layers.iter_mut()) {
            hidden = layer.forward_decode_step(&hidden, plan, state.position, cache)?;
        }
        state.position += 1;
        let post_norm = self.final_norm.forward(&hidden)?;
        let logits = self.lm_head.forward(&post_norm)?;
        Ok((logits, post_norm))
    }

    pub fn decode_tokens_logits_with_hidden(
        &self,
        input_ids: &[u32],
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        let post_norm = self.decode_tokens_hidden(input_ids, plan, state)?;
        let logits = self.logits_from_hidden(&post_norm)?;
        Ok((logits, post_norm))
    }

    pub fn decode_tokens_logits_with_hidden_profiled(
        &self,
        input_ids: &[u32],
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array, DecodeProfileTimings)> {
        let mut profile = DecodeProfileTimings {
            blocks: 1,
            tokens: input_ids.len() as u32,
            ..DecodeProfileTimings::default()
        };
        if input_ids.is_empty() {
            bail!("decode token block cannot be empty");
        }
        if state.layers.len() != self.layers.len() {
            bail!("decode state layer count mismatch");
        }
        let input = input_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, input.len() as i32]);

        let started = Instant::now();
        let mut hidden = self.embeddings.forward(&input_ids)?;
        hidden.eval()?;
        profile.embedding_s += started.elapsed().as_secs_f64();

        for (layer, cache) in self.layers.iter().zip(state.layers.iter_mut()) {
            hidden = layer.forward_decode_tokens_profiled(
                &hidden,
                plan,
                state.position,
                cache,
                &mut profile,
            )?;
        }
        state.position += input.len() as i32;

        let started = Instant::now();
        let post_norm = self.final_norm.forward(&hidden)?;
        post_norm.eval()?;
        profile.final_norm_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let logits = self.logits_from_hidden(&post_norm)?;
        logits.eval()?;
        profile.lm_head_s += started.elapsed().as_secs_f64();

        Ok((logits, post_norm, profile))
    }

    pub fn decode_tokens_hidden(
        &self,
        input_ids: &[u32],
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<mlx_rs::Array> {
        if input_ids.is_empty() {
            bail!("decode token block cannot be empty");
        }
        if state.layers.len() != self.layers.len() {
            bail!("decode state layer count mismatch");
        }
        let input = input_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, input.len() as i32]);
        let mut hidden = self.embeddings.forward(&input_ids)?;
        for (layer, cache) in self.layers.iter().zip(state.layers.iter_mut()) {
            hidden = layer.forward_decode_tokens(&hidden, plan, state.position, cache)?;
        }
        state.position += input.len() as i32;
        self.final_norm.forward(&hidden)
    }

    pub fn logits_from_hidden(&self, hidden: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        self.lm_head.forward(hidden)
    }

    pub fn new_mtp_decode_state(&self) -> Result<MtpDecodeState> {
        if self.mtp.is_none() {
            bail!("model has no MTP sidecar tensors");
        }
        Ok(MtpDecodeState::default())
    }

    pub fn mtp_draft_step(
        &self,
        token_id: u32,
        target_hidden: &mlx_rs::Array,
        plan: &Qwen36Plan,
        state: &mut MtpDecodeState,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array)> {
        let mtp = self
            .mtp
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("model has no MTP sidecar tensors"))?;
        let input = [token_id as i32];
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, 1]);
        let token_embedding = self.embeddings.forward(&input_ids)?;
        let hidden = mtp.forward_decode_step(&token_embedding, target_hidden, plan, state)?;
        let logits = self
            .draft_lm_head
            .as_ref()
            .unwrap_or(&self.lm_head)
            .forward(&hidden)?;
        Ok((logits, hidden))
    }
}

#[cfg(feature = "native-mlx")]
pub(crate) fn array_to_f32_vec(array: &mlx_rs::Array) -> Result<Vec<f32>> {
    let array = array.as_type::<f32>()?;
    array.eval()?;
    Ok(array.as_slice::<f32>().to_vec())
}

#[cfg(feature = "native-mlx")]
fn conv_keep(shape: &[i32], conv_dim: usize) -> Result<usize> {
    if shape.len() != 3 || shape[0] != conv_dim as i32 || shape[2] != 1 || shape[1] < 2 {
        bail!("expected conv1d.weight shape [{conv_dim}, keep + 1, 1], got {shape:?}");
    }
    Ok((shape[1] - 1) as usize)
}

#[cfg(feature = "native-mlx")]
fn conv_weight_at(weight: &[f32], shape: &[i32], channel: usize, k: usize) -> f32 {
    let kernel = shape[1] as usize;
    weight[(channel * kernel + k) * shape[2] as usize]
}

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
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

#[cfg(feature = "native-mlx")]
fn sigmoid_f32(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(feature = "native-mlx")]
fn softplus_f32(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[cfg(feature = "native-mlx")]
fn silu_f32(x: f32) -> f32 {
    x * sigmoid_f32(x)
}

#[cfg(feature = "native-mlx")]
fn norm(
    store: &crate::mlx_backend::MlxWeightStore,
    prefix: &str,
    eps: f32,
) -> Result<crate::mlx_backend::RmsNorm> {
    Ok(crate::mlx_backend::RmsNorm::new(
        store.array(&format!("{prefix}.weight"))?.clone(),
        eps,
    ))
}

#[cfg(feature = "native-mlx")]
fn mtp_norm(
    store: &crate::mlx_backend::MlxWeightStore,
    prefix: &str,
    eps: f32,
) -> Result<crate::mlx_backend::RmsNorm> {
    let raw = store.array(&format!("{prefix}.weight"))?;
    let values = array_to_f32_vec(raw)?;
    let mean = values.iter().copied().sum::<f32>() / values.len().max(1) as f32;
    let weight = if mean < 0.5 {
        raw.add(&mlx_rs::Array::from_slice(&[1.0_f32], &[]))?
    } else {
        raw.clone()
    };
    Ok(crate::mlx_backend::RmsNorm::new(weight, eps))
}

#[cfg(feature = "native-mlx")]
fn quant_i32(
    map: &std::collections::BTreeMap<String, serde_json::Value>,
    key: &str,
    default: i32,
) -> i32 {
    map.get(key)
        .and_then(|value| value.as_i64())
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(default)
}

#[cfg(feature = "native-mlx")]
#[derive(Clone, Copy)]
struct QuantSpec {
    group_size: i32,
    bits: i32,
}

#[cfg(feature = "native-mlx")]
fn quant_spec(
    map: &std::collections::BTreeMap<String, serde_json::Value>,
    prefix: &str,
    default_group_size: i32,
    default_bits: i32,
) -> QuantSpec {
    let Some(value) = map.get(prefix).and_then(|value| value.as_object()) else {
        return QuantSpec {
            group_size: default_group_size,
            bits: default_bits,
        };
    };
    let group_size = value
        .get("group_size")
        .and_then(|value| value.as_i64())
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(default_group_size);
    let bits = value
        .get("bits")
        .and_then(|value| value.as_i64())
        .and_then(|value| i32::try_from(value).ok())
        .unwrap_or(default_bits);
    QuantSpec { group_size, bits }
}

#[cfg(feature = "native-mlx")]
fn draft_lm_head_from_env(
    model: &crate::model::LoadedModel,
    lm_head: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<crate::mlx_backend::QuantizedLinear>> {
    if std::env::var("MTPLX_DISABLE_DRAFT_LM_HEAD")
        .map(|value| {
            matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .unwrap_or(false)
    {
        return Ok(None);
    }
    let contract_spec = model
        .runtime_contract
        .as_ref()
        .and_then(|contract| contract.recommended_draft_lm_head.as_ref());
    let use_contract_spec = env_enabled("MTPLX_ENABLE_DRAFT_LM_HEAD");
    let bits = std::env::var("MTPLX_DRAFT_LM_HEAD_BITS")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .or_else(|| {
            use_contract_spec
                .then(|| contract_spec.map(|spec| spec.bits))
                .flatten()
        })
        .unwrap_or(0);
    if bits <= 0 {
        return Ok(None);
    }
    let group_size = std::env::var("MTPLX_DRAFT_LM_HEAD_GROUP_SIZE")
        .ok()
        .and_then(|value| value.parse::<i32>().ok())
        .or_else(|| {
            use_contract_spec
                .then(|| contract_spec.map(|spec| spec.group_size))
                .flatten()
        })
        .unwrap_or(64);
    lm_head.requantize(group_size, bits).map(Some)
}
