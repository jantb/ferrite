use std::sync::Arc;

use anyhow::Result;

use super::{
    AttentionWeights, FullAttentionWeights, LayerKind, LayerWeights, LinearAttentionWeights,
    MlpWeights, MtpWeights, Qwen36Plan, Qwen36Weights, array_to_f32_vec, env_enabled,
};

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
}

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

#[derive(Clone, Copy)]
struct QuantSpec {
    group_size: i32,
    bits: i32,
}

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
