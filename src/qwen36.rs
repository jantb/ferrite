#![allow(dead_code)]

use std::collections::BTreeSet;
#[cfg(feature = "native-mlx")]
use std::time::Instant;

use anyhow::{Result, bail};
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
mod cache;
#[cfg(feature = "native-mlx")]
mod masks;
#[cfg(feature = "native-mlx")]
pub use cache::{DecodeState, FullAttentionCache, LayerDecodeState, LinearAttentionCache};

#[cfg(feature = "native-mlx")]
mod mtp;
#[cfg(feature = "native-mlx")]
pub use mtp::{MtpDecodeState, MtpWeights};

#[cfg(feature = "native-mlx")]
mod load;

#[cfg(feature = "native-mlx")]
#[derive(Clone, Debug, Default, Serialize)]
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
mod full_attention;
#[cfg(feature = "native-mlx")]
pub use full_attention::{FullAttentionProjection, FullAttentionWeights};
#[cfg(feature = "native-mlx")]
mod linear_attention;
#[cfg(feature = "native-mlx")]
pub use linear_attention::{LinearAttentionProjection, LinearAttentionWeights};
#[cfg(feature = "native-mlx")]
mod layers;
#[cfg(feature = "native-mlx")]
pub use layers::{AttentionWeights, LayerWeights};
#[cfg(feature = "native-mlx")]
mod mlp;
#[cfg(feature = "native-mlx")]
pub use mlp::MlpWeights;

#[cfg(feature = "native-mlx")]
impl Qwen36Weights {
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
                    layers.push(LayerDecodeState::Linear(attn.new_cache(plan)?));
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

    pub fn decode_tokens_last_hidden(
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
        let eval_interval = prefill_layer_eval_interval();
        for (index, (layer, cache)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            hidden = layer.forward_decode_tokens(&hidden, plan, state.position, cache)?;
            if should_eval_prefill_layer(index, eval_interval) {
                hidden.eval()?;
                eval_layer_decode_state(cache)?;
            }
        }
        state.position += input.len() as i32;
        let last = input.len() as i32 - 1;
        self.final_norm
            .forward(&hidden.index((.., last..last + 1, ..)))
    }

    pub fn decode_tokens_cache_only(
        &self,
        input_ids: &[u32],
        plan: &Qwen36Plan,
        state: &mut DecodeState,
    ) -> Result<()> {
        if input_ids.is_empty() {
            bail!("decode token block cannot be empty");
        }
        if state.layers.len() != self.layers.len() {
            bail!("decode state layer count mismatch");
        }
        let input = input_ids.iter().map(|id| *id as i32).collect::<Vec<_>>();
        let input_ids = mlx_rs::Array::from_slice(&input, &[1, input.len() as i32]);
        let mut hidden = self.embeddings.forward(&input_ids)?;
        let eval_interval = prefill_layer_eval_interval();
        for (index, (layer, cache)) in self.layers.iter().zip(state.layers.iter_mut()).enumerate() {
            hidden = layer.forward_decode_tokens(&hidden, plan, state.position, cache)?;
            if should_eval_prefill_layer(index, eval_interval) {
                hidden.eval()?;
                eval_layer_decode_state(cache)?;
            }
        }
        state.position += input.len() as i32;
        Ok(())
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
fn prefill_layer_eval_interval() -> usize {
    std::env::var("FERRITE_PREFILL_EVAL_LAYER_INTERVAL")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(0)
}

#[cfg(feature = "native-mlx")]
fn should_eval_prefill_layer(layer_index: usize, interval: usize) -> bool {
    interval > 0 && (layer_index + 1).is_multiple_of(interval)
}

#[cfg(feature = "native-mlx")]
fn eval_layer_decode_state(cache: &LayerDecodeState) -> Result<()> {
    match cache {
        LayerDecodeState::Full(cache) => {
            if let Some(array) = &cache.k {
                array.eval()?;
            }
            if let Some(array) = &cache.v {
                array.eval()?;
            }
        }
        LayerDecodeState::Linear(cache) => {
            if let Some(array) = &cache.metal_conv_state {
                array.eval()?;
            }
            if let Some(array) = &cache.metal_recurrent_state {
                array.eval()?;
            }
            if let Some(array) = &cache.metal_conv_block_states {
                array.eval()?;
            }
            if let Some(array) = &cache.metal_recurrent_block_states {
                array.eval()?;
            }
        }
    }
    Ok(())
}
