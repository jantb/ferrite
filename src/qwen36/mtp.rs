use anyhow::{Result, bail};
use mlx_rs::ops::concatenate_axis;

use super::{
    AttentionWeights, FullAttentionCache, FullAttentionWeights, LayerWeights, MlpWeights,
    Qwen36Plan, array_to_f32_vec, env_enabled,
};

#[derive(Debug)]
pub struct MtpWeights {
    pub pre_fc_norm_embedding: crate::mlx_backend::RmsNorm,
    pub pre_fc_norm_hidden: crate::mlx_backend::RmsNorm,
    pub fc: crate::mlx_backend::Linear,
    pub layer: LayerWeights,
    pub norm: crate::mlx_backend::RmsNorm,
}

#[derive(Clone, Debug, Default)]
pub struct MtpDecodeState {
    pub position: i32,
    pub cache: FullAttentionCache,
}

impl MtpWeights {
    pub(super) fn from_store(
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
        let down_proj =
            store.quantized_linear(format!("{layer_prefix}.mlp.down_proj"), group_size, bits)?;
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
                mlp: MlpWeights::new(
                    gate_proj,
                    up_proj,
                    down_proj,
                    env_enabled("MTPLX_FUSE_MLP_PROJECTIONS"),
                )?,
            },
            norm: mtp_norm(store, &format!("{prefix}.norm"), eps)?,
        })
    }

    pub(super) fn forward_decode_step(
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
