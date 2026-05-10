use std::sync::Arc;

use anyhow::{Result, bail};
use mlx_rs::ops::indexing::IndexOp;

use super::{LinearAttentionCache, Qwen36Plan, array_to_f32_vec};

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
        self.forward_reference_with_cache_retaining_block_states(x, plan, cache, true)
    }

    pub fn forward_reference_with_cache_without_block_states(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        cache: &mut LinearAttentionCache,
    ) -> Result<mlx_rs::Array> {
        self.forward_reference_with_cache_retaining_block_states(x, plan, cache, false)
    }

    fn forward_reference_with_cache_retaining_block_states(
        &self,
        x: &mlx_rs::Array,
        plan: &Qwen36Plan,
        cache: &mut LinearAttentionCache,
        retain_block_states: bool,
    ) -> Result<mlx_rs::Array> {
        if let Some(metal) = &self.metal {
            return self.forward_metal_with_cache(x, plan, cache, metal, retain_block_states);
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
        retain_block_states: bool,
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
        cache.metal_conv_block_states = if retain_block_states && tokens > 1 {
            Some(conv_states)
        } else {
            None
        };

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
        cache.metal_recurrent_block_states = if retain_block_states && tokens > 1 {
            Some(recurrent_states)
        } else {
            None
        };

        let z = z.reshape(&[batch, tokens, hv, dv])?;
        let gated = self.norm_gate_metal_or_fallback(&out, &z, metal, batch, tokens, hv, dv)?;
        self.out_proj
            .forward(&gated.reshape(&[batch, tokens, hv * dv])?)
    }

    fn norm_gate_metal_or_fallback(
        &self,
        out: &mlx_rs::Array,
        z: &mlx_rs::Array,
        metal: &crate::metal_kernels::LinearGdnKernels,
        batch: i32,
        tokens: i32,
        hv: i32,
        dv: i32,
    ) -> Result<mlx_rs::Array> {
        if self.norm.weight.shape() == [dv] {
            return metal.norm_gate(
                out,
                z,
                &self.norm.weight,
                self.norm.eps,
                batch,
                tokens,
                hv,
                dv,
            );
        }

        let gate = mlx_rs::nn::silu(z)?;
        let normed = self.norm.forward(out)?;
        Ok(&gate * &normed)
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
