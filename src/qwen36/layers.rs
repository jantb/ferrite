use std::time::Instant;

use super::{
    DecodeProfileTimings, FullAttentionWeights, LayerDecodeState, LinearAttentionWeights,
    MlpWeights, Qwen36Plan,
};
use anyhow::{Result, bail};

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
