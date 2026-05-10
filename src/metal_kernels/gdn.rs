use anyhow::{Result, anyhow, bail};
use mlx_rs::{Array, Stream};

use super::{MetalKernel, OutputSpec, TemplateArg};

#[derive(Debug)]
pub struct LinearGdnKernels {
    conv1d: MetalKernel,
    gated_delta_inline: MetalKernel,
    norm_gate: MetalKernel,
}

impl LinearGdnKernels {
    pub fn new() -> Result<Self> {
        Ok(Self {
            conv1d: MetalKernel::new(
                "mtplx_rs_linear_conv1d",
                &["qkv", "base_conv_state", "conv_weight", "T"],
                &["conv_out", "conv_states"],
                LINEAR_CONV1D_SOURCE,
                "",
            )?,
            gated_delta_inline: MetalKernel::new(
                "mtplx_rs_linear_gated_delta_inline_g",
                &["conv_out", "a", "b", "A_log", "dt_bias", "state_in", "T"],
                &["y", "states"],
                LINEAR_GATED_DELTA_INLINE_SOURCE,
                "",
            )?,
            norm_gate: MetalKernel::new(
                "mtplx_rs_linear_norm_gate",
                &["y", "z", "norm_weight", "eps"],
                &["out"],
                LINEAR_NORM_GATE_SOURCE,
                "",
            )?,
        })
    }

    pub fn conv1d_silu(
        &self,
        qkv: &Array,
        base_conv_state: &Array,
        conv_weight: &Array,
        batch: i32,
        tokens: i32,
        conv_dim: i32,
        keep: i32,
    ) -> Result<(Array, Array)> {
        let t = Array::from_int(tokens);
        let outputs = self.conv1d.apply(
            &[qkv, base_conv_state, conv_weight, &t],
            &[
                OutputSpec {
                    shape: &[batch, tokens, conv_dim],
                    dtype: qkv.dtype(),
                },
                OutputSpec {
                    shape: &[batch, tokens, keep, conv_dim],
                    dtype: qkv.dtype(),
                },
            ],
            &[
                TemplateArg::Dtype("InT", qkv.dtype()),
                TemplateArg::Int("Keep", keep),
                TemplateArg::Int("ConvDim", conv_dim),
            ],
            (conv_dim, batch, 1),
            (256, 1, 1),
            &Stream::gpu(),
        )?;
        let [raw_conv, conv_states]: [Array; 2] = outputs
            .try_into()
            .map_err(|_| anyhow!("linear conv1d kernel returned wrong output count"))?;
        Ok((mlx_rs::nn::silu(&raw_conv)?, conv_states))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gated_delta_inline(
        &self,
        conv_out: &Array,
        a: &Array,
        b: &Array,
        a_log: &Array,
        dt_bias: &Array,
        state_in: &Array,
        batch: i32,
        tokens: i32,
        hk: i32,
        hv: i32,
        dk: i32,
        dv: i32,
    ) -> Result<(Array, Array)> {
        if dk % 32 != 0 {
            bail!("linear GDN Metal kernel requires Dk to be divisible by 32, got {dk}");
        }
        let key_dim = hk * dk;
        let conv_dim = key_dim * 2 + hv * dv;
        let tgy = if dv % 32 == 0 {
            32
        } else if dv % 16 == 0 {
            16
        } else if dv % 8 == 0 {
            8
        } else {
            4
        };
        let t = Array::from_int(tokens);
        let outputs = self.gated_delta_inline.apply(
            &[conv_out, a, b, a_log, dt_bias, state_in, &t],
            &[
                OutputSpec {
                    shape: &[batch, tokens, hv, dv],
                    dtype: conv_out.dtype(),
                },
                OutputSpec {
                    shape: &[batch, tokens, hv, dv, dk],
                    dtype: state_in.dtype(),
                },
            ],
            &[
                TemplateArg::Dtype("InT", conv_out.dtype()),
                TemplateArg::Dtype("StT", state_in.dtype()),
                TemplateArg::Int("Dk", dk),
                TemplateArg::Int("Dv", dv),
                TemplateArg::Int("Hk", hk),
                TemplateArg::Int("Hv", hv),
                TemplateArg::Int("KeyDim", key_dim),
                TemplateArg::Int("ConvDim", conv_dim),
            ],
            (32, dv, batch * hv),
            (32, tgy, 1),
            &Stream::gpu(),
        )?;
        let [y, states]: [Array; 2] = outputs
            .try_into()
            .map_err(|_| anyhow!("linear GDN kernel returned wrong output count"))?;
        Ok((y, states))
    }

    pub fn norm_gate(
        &self,
        y: &Array,
        z: &Array,
        norm_weight: &Array,
        eps: f32,
        batch: i32,
        tokens: i32,
        hv: i32,
        dv: i32,
    ) -> Result<Array> {
        let eps = Array::from_f32(eps);
        let outputs = self.norm_gate.apply(
            &[y, z, norm_weight, &eps],
            &[OutputSpec {
                shape: &[batch, tokens, hv, dv],
                dtype: y.dtype(),
            }],
            &[
                TemplateArg::Dtype("InT", y.dtype()),
                TemplateArg::Dtype("WtT", norm_weight.dtype()),
                TemplateArg::Int("Dv", dv),
            ],
            (32, batch * tokens * hv, 1),
            (32, 1, 1),
            &Stream::gpu(),
        )?;
        let [out]: [Array; 1] = outputs
            .try_into()
            .map_err(|_| anyhow!("linear norm/gate kernel returned wrong output count"))?;
        Ok(out)
    }
}

const LINEAR_CONV1D_SOURCE: &str = r#"
    auto c_idx = thread_position_in_grid.x;
    auto b_idx = thread_position_in_grid.y;

    if (c_idx >= ConvDim) {
      return;
    }

    for (int t = 0; t < T; ++t) {
      auto parent_idx = t - 1;

      float acc = 0.0f;
      for (int k = 0; k < Keep; ++k) {
        float x;
        if (parent_idx < 0) {
          x = static_cast<float>(
            base_conv_state[(b_idx * Keep + k) * ConvDim + c_idx]
          );
        } else {
          x = static_cast<float>(
            conv_states[
              (((b_idx * T + parent_idx) * Keep + k) * ConvDim) + c_idx
            ]
          );
        }
        auto w = static_cast<float>(conv_weight[c_idx * (Keep + 1) + k]);
        acc += x * w;
      }

      auto qkv_t = qkv + (b_idx * T + t) * ConvDim;
      acc += static_cast<float>(qkv_t[c_idx])
        * static_cast<float>(conv_weight[c_idx * (Keep + 1) + Keep]);

      conv_out[(b_idx * T + t) * ConvDim + c_idx] =
        static_cast<InT>(acc);

      for (int k = 0; k < Keep; ++k) {
        InT value;
        if (k + 1 < Keep) {
          if (parent_idx < 0) {
            value = base_conv_state[(b_idx * Keep + k + 1) * ConvDim + c_idx];
          } else {
            value = conv_states[
              (((b_idx * T + parent_idx) * Keep + k + 1) * ConvDim) + c_idx
            ];
          }
        } else {
          value = qkv_t[c_idx];
        }
        conv_states[
          (((b_idx * T + t) * Keep + k) * ConvDim) + c_idx
        ] = value;
      }
    }
"#;

const LINEAR_GATED_DELTA_INLINE_SOURCE: &str = r#"
    auto n = thread_position_in_grid.z;
    auto b_idx = n / Hv;
    auto hv_idx = n % Hv;
    auto hk_idx = hv_idx / (Hv / Hk);
    constexpr int n_per_t = Dk / 32;

    auto dk_idx = thread_position_in_threadgroup.x;
    auto local_dv_idx = thread_position_in_threadgroup.y;
    auto dv_idx = thread_position_in_grid.y;
    float inv_scale = 1.0f / metal::sqrt(float(Dk));
    float q_scale = inv_scale * inv_scale;
    float k_scale = static_cast<float>(static_cast<InT>(inv_scale));
    threadgroup float q_shared[Dk];
    threadgroup float k_shared[Dk];
    threadgroup float g_shared;
    threadgroup float beta_shared;

    for (int t = 0; t < T; ++t) {
      auto parent_idx = t - 1;

      const device StT* parent_state;
      if (parent_idx < 0) {
        parent_state = state_in + (n * Dv + dv_idx) * Dk;
      } else {
        parent_state = states
          + (((b_idx * T + parent_idx) * Hv + hv_idx) * Dv + dv_idx) * Dk;
      }

      float state[n_per_t];
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state[i] = static_cast<float>(parent_state[s_idx]);
      }

      auto conv_t = conv_out + (b_idx * T + t) * ConvDim;
      auto q_t = conv_t + hk_idx * Dk;
      auto k_t = conv_t + KeyDim + hk_idx * Dk;
      auto v_t = conv_t + 2 * KeyDim + hv_idx * Dv;
      auto a_t = a + (b_idx * T + t) * Hv;
      auto b_t = b + (b_idx * T + t) * Hv;

      if (dk_idx == 0 && local_dv_idx == 0) {
        InT b_val = b_t[hv_idx];
        auto beta_y = 1 / (1 + metal::exp(metal::abs(b_val)));
        InT beta_val = (b_val < InT(0)) ? beta_y : 1 - beta_y;

        InT a_val = a_t[hv_idx] + dt_bias[hv_idx];
        constexpr InT inf = metal::numeric_limits<InT>::infinity();
        InT maxval = metal::max(a_val, InT(0));
        InT minval = metal::min(a_val, InT(0));
        InT softplus_val = (minval == -inf || maxval == inf)
          ? maxval
          : (maxval + log1p(metal::exp(minval - maxval)));
        float decay_a = metal::exp(float(A_log[hv_idx]));
        beta_shared = static_cast<float>(beta_val);
        g_shared = metal::exp(-decay_a * float(softplus_val));
      }

      if (local_dv_idx == 0) {
        float q_sum = 0.0f;
        float k_sum = 0.0f;
        float q_raw[n_per_t];
        float k_raw[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          q_raw[i] = static_cast<float>(q_t[s_idx]);
          k_raw[i] = static_cast<float>(k_t[s_idx]);
          q_sum += q_raw[i] * q_raw[i];
          k_sum += k_raw[i] * k_raw[i];
        }
        q_sum = simd_sum(q_sum);
        k_sum = simd_sum(k_sum);
        float q_inv = metal::precise::rsqrt(q_sum / float(Dk) + 1.0e-6f);
        float k_inv = metal::precise::rsqrt(k_sum / float(Dk) + 1.0e-6f);

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          auto q_norm = static_cast<InT>(q_raw[i] * q_inv);
          auto k_norm = static_cast<InT>(k_raw[i] * k_inv);
          q_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(q_norm) * q_scale));
          k_shared[s_idx] =
            static_cast<float>(static_cast<InT>(static_cast<float>(k_norm) * k_scale));
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);

      float kv_mem = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        auto k_val = k_shared[s_idx];
        state[i] = state[i] * g_shared;
        kv_mem += state[i] * k_val;
      }
      kv_mem = simd_sum(kv_mem);

      auto delta = (static_cast<float>(v_t[dv_idx]) - kv_mem)
        * beta_shared;

      float out = 0.0f;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        auto k_val = k_shared[s_idx];
        auto q_val = q_shared[s_idx];
        state[i] = state[i] + k_val * delta;
        out += state[i] * q_val;
      }
      out = simd_sum(out);

      auto y_t = y + ((b_idx * T + t) * Hv + hv_idx) * Dv;
      if (thread_index_in_simdgroup == 0) {
        y_t[dv_idx] = static_cast<InT>(out);
      }

      auto state_t = states
        + (((b_idx * T + t) * Hv + hv_idx) * Dv + dv_idx) * Dk;
      for (int i = 0; i < n_per_t; ++i) {
        auto s_idx = n_per_t * dk_idx + i;
        state_t[s_idx] = static_cast<StT>(state[i]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
"#;

const LINEAR_NORM_GATE_SOURCE: &str = r#"
    auto lane = thread_index_in_simdgroup;
    auto row = thread_position_in_grid.y;
    auto base = row * Dv;

    float sum = 0.0f;
    for (int i = lane; i < Dv; i += 32) {
      float value = static_cast<float>(y[base + i]);
      sum += value * value;
    }
    sum = simd_sum(sum);
    float inv = metal::precise::rsqrt(sum / float(Dv) + float(eps));

    for (int i = lane; i < Dv; i += 32) {
      float z_value = static_cast<float>(z[base + i]);
      float gate = z_value / (1.0f + metal::exp(-z_value));
      float normalized = static_cast<float>(y[base + i])
        * inv
        * static_cast<float>(norm_weight[i]);
      out[base + i] = static_cast<InT>(gate * normalized);
    }
"#;
