#![allow(dead_code)]

use std::cell::RefCell;
use std::ffi::{CString, c_char};
use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, Stream};

const SUCCESS: i32 = 0;

pub fn metal_is_available() -> bool {
    let mut available = false;
    let status = unsafe { mlx_sys::mlx_metal_is_available(&mut available as *mut _) };
    status == SUCCESS && available
}

pub fn small_m_qmm4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_SMALL_M_QMM4", false))
}

pub fn gate_up_swiglu_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_GATE_UP_SWIGLU_QMV4", false))
}

pub fn gate_up_swiglu_qmv4_activation(
    x: &Array,
    gate: &crate::mlx_backend::QuantizedLinear,
    up: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !gate_up_swiglu_qmv4_is_eligible(x, gate, up) {
        return Ok(None);
    }

    let shape = x.shape().to_vec();
    let m = shape[shape.len() - 2];
    let k = shape[shape.len() - 1];
    let n = gate.weight.shape()[0];
    let x2 = x.reshape(&[m, k])?;
    let m_arg = Array::from_int(m);
    let k_arg = Array::from_int(k);
    let n_arg = Array::from_int(n);
    let grid_y = 2 * ((n + 7) / 8);
    let outputs = with_gate_up_swiglu_qmv4_kernel(x.dtype(), |kernel| {
        kernel.apply(
            &[
                &x2,
                &gate.weight,
                &gate.scales,
                &gate.biases,
                &up.weight,
                &up.scales,
                &up.biases,
                &m_arg,
                &k_arg,
                &n_arg,
            ],
            &[OutputSpec {
                shape: &[m, n],
                dtype: x.dtype(),
            }],
            &[
                TemplateArg::Dtype("T", x.dtype()),
                TemplateArg::Int("GS", gate.group_size),
            ],
            (32, grid_y, 1),
            (32, 2, 1),
            &Stream::gpu(),
        )
    })?;
    let [y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("gate/up SwiGLU qmv4 kernel returned wrong output count"))?;
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    }
    Ok(Some(y.reshape(&out_shape)?))
}

fn gate_up_swiglu_qmv4_is_eligible(
    x: &Array,
    gate: &crate::mlx_backend::QuantizedLinear,
    up: &crate::mlx_backend::QuantizedLinear,
) -> bool {
    if !gate_up_swiglu_qmv4_enabled() || !metal_is_available() {
        return false;
    }
    if gate.bits != 4 || up.bits != 4 {
        return false;
    }
    if gate.group_size != up.group_size || !matches!(gate.group_size, 32 | 64 | 128) {
        return false;
    }
    if gate.bias.is_some() || up.bias.is_some() {
        return false;
    }
    if !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        return false;
    }
    if gate.scales.dtype() != x.dtype()
        || gate.biases.dtype() != x.dtype()
        || up.scales.dtype() != x.dtype()
        || up.biases.dtype() != x.dtype()
    {
        return false;
    }
    let shape = x.shape();
    if shape.len() < 2 {
        return false;
    }
    let m = shape[shape.len() - 2];
    if !(2..=6).contains(&m) {
        return false;
    }
    let leading_batch = shape[..shape.len() - 2].iter().copied().product::<i32>();
    if leading_batch != 1 {
        return false;
    }
    let k = shape[shape.len() - 1];
    let gate_weight_shape = gate.weight.shape();
    if gate_weight_shape.len() != 2 {
        return false;
    }
    let n = gate_weight_shape[0];
    gate.weight.shape() == up.weight.shape()
        && gate.scales.shape() == up.scales.shape()
        && gate.biases.shape() == up.biases.shape()
        && k == gate_weight_shape[1] * 8
        && k % 512 == 0
        && n % 8 == 0
}

pub fn small_m_qmm4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !small_m_qmm4_is_eligible(x, linear) {
        return Ok(None);
    }

    let shape = x.shape().to_vec();
    let m = shape[shape.len() - 2];
    let k = shape[shape.len() - 1];
    let n = linear.weight.shape()[0];
    let x2 = x.reshape(&[m, k])?;
    let x8 = if m < 8 {
        let pad = mlx_rs::ops::zeros_dtype(&[8 - m, k], x.dtype())?;
        mlx_rs::ops::concatenate_axis(&[x2, pad], 0)?
    } else {
        x2
    };
    let m_arg = Array::from_int(m);
    let k_arg = Array::from_int(k);
    let n_arg = Array::from_int(n);
    let outputs = with_small_m_qmm4_kernel(x.dtype(), |kernel| {
        kernel.apply(
            &[
                &x8,
                &linear.weight,
                &linear.scales,
                &linear.biases,
                &m_arg,
                &k_arg,
                &n_arg,
            ],
            &[OutputSpec {
                shape: &[8, n],
                dtype: x.dtype(),
            }],
            &[
                TemplateArg::Dtype("T", x.dtype()),
                TemplateArg::Int("GS", linear.group_size),
            ],
            (64, n / 32, 1),
            (64, 1, 1),
            &Stream::gpu(),
        )
    })?;
    let [y8]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("small-m qmm4 kernel returned wrong output count"))?;
    let mut y = y8.index((0..m, ..));
    if let Some(bias) = &linear.bias {
        y = y.add(bias)?;
    }
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    }
    Ok(Some(y.reshape(&out_shape)?))
}

fn small_m_qmm4_is_eligible(x: &Array, linear: &crate::mlx_backend::QuantizedLinear) -> bool {
    if !small_m_qmm4_enabled() || !metal_is_available() {
        return false;
    }
    if linear.bits != 4 || !matches!(linear.group_size, 32 | 64 | 128) {
        return false;
    }
    if !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        return false;
    }
    if linear.scales.dtype() != x.dtype() || linear.biases.dtype() != x.dtype() {
        return false;
    }
    let shape = x.shape();
    if shape.len() < 2 {
        return false;
    }
    let m = shape[shape.len() - 2];
    if !(2..=8).contains(&m) {
        return false;
    }
    let leading_batch = shape[..shape.len() - 2].iter().copied().product::<i32>();
    if leading_batch != 1 {
        return false;
    }
    let k = shape[shape.len() - 1];
    let weight_shape = linear.weight.shape();
    if weight_shape.len() != 2 {
        return false;
    }
    let n = weight_shape[0];
    k == weight_shape[1] * 8 && k % 32 == 0 && n % 32 == 0
}

pub fn multi3_qmv4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !multi3_qmv4_is_eligible(x, linear) {
        return Ok(None);
    }

    let shape = x.shape().to_vec();
    let k = *shape.last().unwrap_or(&0);
    let n = linear.weight.shape()[0];
    let x2 = x.reshape(&[3, k])?;
    let k_arg = Array::from_int(k);
    let n_arg = Array::from_int(n);
    let outputs = with_multi3_qmv4_kernel(x.dtype(), |kernel| {
        kernel.apply(
            &[
                &x2,
                &linear.weight,
                &linear.scales,
                &linear.biases,
                &k_arg,
                &n_arg,
            ],
            &[OutputSpec {
                shape: &[3, n],
                dtype: x.dtype(),
            }],
            &[
                TemplateArg::Dtype("T", x.dtype()),
                TemplateArg::Int("GS", linear.group_size),
            ],
            (32, 2 * (n / 8), 1),
            (32, 2, 1),
            &Stream::gpu(),
        )
    })?;
    let [mut y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("multi3 qmv4 kernel returned wrong output count"))?;
    if let Some(bias) = &linear.bias {
        y = y.add(bias)?;
    }
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    }
    Ok(Some(y.reshape(&out_shape)?))
}

fn multi3_qmv4_is_eligible(x: &Array, linear: &crate::mlx_backend::QuantizedLinear) -> bool {
    if !multi3_qmv4_enabled() || !metal_is_available() {
        return false;
    }
    if linear.bits != 4 || !matches!(linear.group_size, 32 | 64 | 128) {
        return false;
    }
    if !matches!(x.dtype(), Dtype::Bfloat16 | Dtype::Float16) {
        return false;
    }
    if linear.scales.dtype() != x.dtype() || linear.biases.dtype() != x.dtype() {
        return false;
    }
    let shape = x.shape();
    if shape.len() < 2 || shape[shape.len() - 2] != 3 {
        return false;
    }
    let leading_batch = shape[..shape.len() - 2].iter().copied().product::<i32>();
    if leading_batch != 1 {
        return false;
    }
    let k = shape[shape.len() - 1];
    let weight_shape = linear.weight.shape();
    if weight_shape.len() != 2 {
        return false;
    }
    let n = weight_shape[0];
    k == weight_shape[1] * 8 && k % 512 == 0 && n % 8 == 0
}

fn multi3_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_MULTI3_QMV4", false))
}

thread_local! {
    static GATE_UP_SWIGLU_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static GATE_UP_SWIGLU_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMM4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMM4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static MULTI3_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static MULTI3_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

fn with_gate_up_swiglu_qmv4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => {
            GATE_UP_SWIGLU_QMV4_BF16.with(|slot| with_cached_gate_up_kernel(slot, dtype, f))
        }
        Dtype::Float16 => {
            GATE_UP_SWIGLU_QMV4_F16.with(|slot| with_cached_gate_up_kernel(slot, dtype, f))
        }
        other => bail!("gate/up SwiGLU qmv4 does not support dtype {other:?}"),
    }
}

fn with_cached_gate_up_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "mtplx_rs_gate_up_swiglu_qmv4_bf16",
            Dtype::Float16 => "mtplx_rs_gate_up_swiglu_qmv4_f16",
            other => bail!("gate/up SwiGLU qmv4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &[
                "x",
                "gate_w",
                "gate_scales",
                "gate_biases",
                "up_w",
                "up_scales",
                "up_biases",
                "M_size",
                "K_size",
                "N_size",
            ],
            &["y"],
            GATE_UP_SWIGLU_QMV4_SOURCE,
            GATE_UP_SWIGLU_QMV4_HEADER,
        )?);
    }
    let kernel = slot.borrow();
    f(kernel
        .as_ref()
        .expect("gate/up SwiGLU qmv4 kernel initialized"))
}

fn with_small_m_qmm4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => {
            SMALL_M_QMM4_BF16.with(|slot| with_cached_small_m_kernel(slot, dtype, f))
        }
        Dtype::Float16 => SMALL_M_QMM4_F16.with(|slot| with_cached_small_m_kernel(slot, dtype, f)),
        other => bail!("small-m qmm4 does not support dtype {other:?}"),
    }
}

fn with_cached_small_m_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "mtplx_rs_small_m_qmm4_bf16",
            Dtype::Float16 => "mtplx_rs_small_m_qmm4_f16",
            other => bail!("small-m qmm4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
            &["y"],
            SMALL_M_QMM4_SOURCE,
            "",
        )?);
    }
    let kernel = slot.borrow();
    f(kernel.as_ref().expect("small-m qmm4 kernel initialized"))
}

fn with_multi3_qmv4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => MULTI3_QMV4_BF16.with(|slot| with_cached_multi3_kernel(slot, dtype, f)),
        Dtype::Float16 => MULTI3_QMV4_F16.with(|slot| with_cached_multi3_kernel(slot, dtype, f)),
        other => bail!("multi3 qmv4 does not support dtype {other:?}"),
    }
}

fn with_cached_multi3_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "mtplx_rs_multi3_qmv4_bf16",
            Dtype::Float16 => "mtplx_rs_multi3_qmv4_f16",
            other => bail!("multi3 qmv4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &["x", "w", "scales", "biases", "K_size", "N_size"],
            &["y"],
            MULTI3_QMV4_SOURCE,
            MULTI3_QMV4_HEADER,
        )?);
    }
    let kernel = slot.borrow();
    f(kernel.as_ref().expect("multi3 qmv4 kernel initialized"))
}

fn env_flag(name: &str, default: bool) -> bool {
    std::env::var(name)
        .ok()
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            matches!(value.as_str(), "1" | "true" | "yes" | "on")
                || (!matches!(value.as_str(), "0" | "false" | "no" | "off") && default)
        })
        .unwrap_or(default)
}

#[derive(Debug)]
pub struct MetalKernel {
    raw: mlx_sys::mlx_fast_metal_kernel,
}

impl MetalKernel {
    pub fn new(
        name: &str,
        input_names: &[&str],
        output_names: &[&str],
        source: &str,
        header: &str,
    ) -> Result<Self> {
        let name = CString::new(name)?;
        let source = CString::new(source)?;
        let header = CString::new(header)?;
        let inputs = CStringVec::new(input_names)?;
        let outputs = CStringVec::new(output_names)?;

        let raw = unsafe {
            mlx_sys::mlx_fast_metal_kernel_new(
                name.as_ptr(),
                inputs.raw(),
                outputs.raw(),
                source.as_ptr(),
                header.as_ptr(),
                true,
                false,
            )
        };
        if raw.ctx.is_null() {
            bail!("failed to create MLX Metal kernel");
        }
        Ok(Self { raw })
    }

    pub fn apply(
        &self,
        inputs: &[&Array],
        output_specs: &[OutputSpec<'_>],
        template_args: &[TemplateArg<'_>],
        grid: (i32, i32, i32),
        thread_group: (i32, i32, i32),
        stream: &Stream,
    ) -> Result<Vec<Array>> {
        let input_vec = ArrayVec::new(inputs)?;
        let config = MetalKernelConfig::new()?;
        for spec in output_specs {
            config.add_output_arg(spec.shape, spec.dtype)?;
        }
        for arg in template_args {
            config.add_template_arg(arg)?;
        }
        config.set_grid(grid)?;
        config.set_thread_group(thread_group)?;

        let mut outputs = unsafe { mlx_sys::mlx_vector_array_new() };
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_apply(
                &mut outputs as *mut _,
                self.raw,
                input_vec.raw,
                config.raw,
                stream.as_ptr(),
            )
        };
        if status != SUCCESS {
            unsafe {
                mlx_sys::mlx_vector_array_free(outputs);
            }
            bail!("MLX Metal kernel apply failed with status {status}");
        }
        let outputs = ArrayVec { raw: outputs };
        outputs.into_arrays()
    }
}

impl Drop for MetalKernel {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_free(self.raw);
        }
    }
}

pub struct OutputSpec<'a> {
    pub shape: &'a [i32],
    pub dtype: Dtype,
}

pub enum TemplateArg<'a> {
    Dtype(&'a str, Dtype),
    Int(&'a str, i32),
    Bool(&'a str, bool),
}

#[derive(Debug)]
pub struct LinearGdnKernels {
    conv1d: MetalKernel,
    gated_delta_inline: MetalKernel,
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
}

const MULTI3_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"#;

const GATE_UP_SWIGLU_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
    constant constexpr int NUM_SIMDGROUPS = 2;
    constant constexpr int BN = RESULTS_PER_SIMDGROUP * NUM_SIMDGROUPS;
    constant constexpr int MAX_M = 6;

    template <typename T>
    inline T sigmoid_mlx_exact(T x) {
      auto y = 1 / (1 + metal::exp(metal::abs(x)));
      return (x < T(0)) ? y : 1 - y;
    }

    template <typename T>
    inline T swiglu_mlx_exact(T gate, T up) {
      T silu = gate * sigmoid_mlx_exact<T>(gate);
      return T(silu * up);
    }

    template <typename T>
    inline float load_vector4_exact(const device T* x, thread float* x_thread) {
      float sum = 0.0f;
      for (int i = 0; i < VALUES_PER_THREAD; i += 4) {
        sum += x[i] + x[i + 1] + x[i + 2] + x[i + 3];
        x_thread[i] = x[i];
        x_thread[i + 1] = x[i + 1] / 16.0f;
        x_thread[i + 2] = x[i + 2] / 256.0f;
        x_thread[i + 3] = x[i + 3] / 4096.0f;
      }
      return sum;
    }

    inline float qdot4_exact(
        const device uint8_t* w,
        const thread float* x_thread,
        float scale,
        float bias,
        float sum) {
      const device uint16_t* ws = (const device uint16_t*)w;
      float accum = 0.0f;
      for (int i = 0; i < (VALUES_PER_THREAD / 4); ++i) {
        uint16_t packed = ws[i];
        accum +=
          x_thread[4 * i] * float(packed & 0x000f) +
          x_thread[4 * i + 1] * float(packed & 0x00f0) +
          x_thread[4 * i + 2] * float(packed & 0x0f00) +
          x_thread[4 * i + 3] * float(packed & 0xf000);
      }
      return scale * accum + sum * bias;
    }
"#;

const GATE_UP_SWIGLU_QMV4_SOURCE: &str = r#"
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int M = int(M_size);
    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* gate_w_base =
      (const device uint8_t*)gate_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device uint8_t* up_w_base =
      (const device uint8_t*)up_w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* gate_scales_base =
      gate_scales + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* gate_biases_base =
      gate_biases + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_scales_base =
      up_scales + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* up_biases_base =
      up_biases + out_row * in_vec_size_g
      + int(simd_lid) / SCALE_STEP_PER_THREAD;

    float gate_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float up_result[MAX_M][RESULTS_PER_SIMDGROUP];
    float x_thread[MAX_M][VALUES_PER_THREAD];
    float x_sum[MAX_M];

    for (int m = 0; m < MAX_M; ++m) {
      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        gate_result[m][row] = 0.0f;
        up_result[m][row] = 0.0f;
      }
    }

    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      for (int m = 0; m < MAX_M; ++m) {
        if (m < M) {
          const device T* x_m =
            x + m * K + k_block + int(simd_lid) * VALUES_PER_THREAD;
          x_sum[m] = load_vector4_exact<T>(x_m, x_thread[m]);
        }
      }

      const device uint8_t* gate_w_block =
        gate_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device uint8_t* up_w_block =
        up_w_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* gate_scales_block = gate_scales_base + k_block / GS;
      const device T* gate_biases_block = gate_biases_base + k_block / GS;
      const device T* up_scales_block = up_scales_base + k_block / GS;
      const device T* up_biases_block = up_biases_base + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* gate_w_row =
            gate_w_block + row * in_vec_size_w;
          const device uint8_t* up_w_row =
            up_w_block + row * in_vec_size_w;
          const device T* gate_sc_row =
            gate_scales_block + row * in_vec_size_g;
          const device T* gate_bs_row =
            gate_biases_block + row * in_vec_size_g;
          const device T* up_sc_row =
            up_scales_block + row * in_vec_size_g;
          const device T* up_bs_row =
            up_biases_block + row * in_vec_size_g;
          float gate_scale = float(gate_sc_row[0]);
          float gate_bias = float(gate_bs_row[0]);
          float up_scale = float(up_sc_row[0]);
          float up_bias = float(up_bs_row[0]);

          for (int m = 0; m < MAX_M; ++m) {
            if (m < M) {
              gate_result[m][row] += qdot4_exact(
                gate_w_row, x_thread[m], gate_scale, gate_bias, x_sum[m]
              );
              up_result[m][row] += qdot4_exact(
                up_w_row, x_thread[m], up_scale, up_bias, x_sum[m]
              );
            }
          }
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        for (int m = 0; m < MAX_M; ++m) {
          if (m < M) {
            float gate_sum = simd_sum(gate_result[m][row]);
            float up_sum = simd_sum(up_result[m][row]);
            if (simd_lid == 0) {
              T gate_value = T(gate_sum);
              T up_value = T(up_sum);
              y[m * N + n] = swiglu_mlx_exact<T>(gate_value, up_value);
            }
          }
        }
      }
    }
"#;

const SMALL_M_QMM4_SOURCE: &str = r#"
    using namespace metal;
    constexpr int BM = 8;
    constexpr int BN = 32;
    constexpr int BK = 32;
    constexpr int BK_SUB = 8;

    uint tid   = thread_position_in_threadgroup.x;
    uint sg_id = tid / 32;
    uint tg_n  = threadgroup_position_in_grid.y;

    int K = int(K_size);
    int N = int(N_size);
    int K_by_8  = K / 8;
    int K_by_gs = K / GS;
    int n0 = int(tg_n) * BN;

    threadgroup T B_tile[BK * BN];

    simdgroup_matrix<T, 8, 8> a, b_L, b_R;
    simdgroup_matrix<float, 8, 8> c_L =
      simdgroup_matrix<float, 8, 8>(0.0f);
    simdgroup_matrix<float, 8, 8> c_R =
      simdgroup_matrix<float, 8, 8>(0.0f);

    int t_a = int(tid);
    int t_b = int(tid) + 64;
    int dq_k_a = t_a / BN, dq_n_a = t_a % BN;
    int dq_k_b = t_b / BN, dq_n_b = t_b % BN;
    int sg_n_off = int(sg_id) * 16;

    for (int k0 = 0; k0 < K; k0 += BK) {
        {
            int n_global = n0 + dq_n_a;
            int k_base = k0 + dq_k_a * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_a * 8 + ki) * BN + dq_n_a] =
                  T(float(nib) * s + b);
            }
        }
        {
            int n_global = n0 + dq_n_b;
            int k_base = k0 + dq_k_b * 8;
            uint32_t packed = w_q[n_global * K_by_8 + (k_base >> 3)];
            float s = float(scales[n_global * K_by_gs + (k_base / GS)]);
            float b = float(biases[n_global * K_by_gs + (k_base / GS)]);
            for (int ki = 0; ki < 8; ++ki) {
                uint32_t nib = (packed >> (ki * 4)) & 0xFu;
                B_tile[(dq_k_b * 8 + ki) * BN + dq_n_b] =
                  T(float(nib) * s + b);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int ks = 0; ks < BK / BK_SUB; ++ks) {
            simdgroup_load(a, x + k0 + ks * BK_SUB, K);
            simdgroup_load(
              b_L, B_tile + ks * BK_SUB * BN + sg_n_off, BN
            );
            simdgroup_load(
              b_R, B_tile + ks * BK_SUB * BN + sg_n_off + 8, BN
            );
            simdgroup_multiply_accumulate(c_L, a, b_L, c_L);
            simdgroup_multiply_accumulate(c_R, a, b_R, c_R);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_matrix<T, 8, 8> c_L_T, c_R_T;
    c_L_T.thread_elements()[0] = T(c_L.thread_elements()[0]);
    c_L_T.thread_elements()[1] = T(c_L.thread_elements()[1]);
    c_R_T.thread_elements()[0] = T(c_R.thread_elements()[0]);
    c_R_T.thread_elements()[1] = T(c_R.thread_elements()[1]);
    simdgroup_store(c_L_T, y + n0 + sg_n_off, N);
    simdgroup_store(c_R_T, y + n0 + sg_n_off + 8, N);
"#;

const MULTI3_QMV4_SOURCE: &str = r#"
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int out_row = int(n_tile) * BN + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* ws_base =
      (const device uint8_t*)w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
      scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
      biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;

    const device T* x0_base = x + int(simd_lid) * VALUES_PER_THREAD;
    const device T* x1_base = x + K + int(simd_lid) * VALUES_PER_THREAD;
    const device T* x2_base = x + 2 * K + int(simd_lid) * VALUES_PER_THREAD;

    float result0[RESULTS_PER_SIMDGROUP] = {0.0f};
    float result1[RESULTS_PER_SIMDGROUP] = {0.0f};
    float result2[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x0_thread[VALUES_PER_THREAD];
    float x1_thread[VALUES_PER_THREAD];
    float x2_thread[VALUES_PER_THREAD];

    const device uint8_t* ws = ws_base;
    const device T* sc = scales_base;
    const device T* bs = biases_base;
    const device T* x0 = x0_base;
    const device T* x1 = x1_base;
    const device T* x2 = x2_base;

    for (int k = 0; k < K; k += BLOCK_SIZE) {
      float sum0 = load_vector4_exact<T>(x0, x0_thread);
      float sum1 = load_vector4_exact<T>(x1, x1_thread);
      float sum2 = load_vector4_exact<T>(x2, x2_thread);

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* wl = ws + row * in_vec_size_w;
          const device T* sl = sc + row * in_vec_size_g;
          const device T* bl = bs + row * in_vec_size_g;
          float s = float(sl[0]);
          float b = float(bl[0]);
          result0[row] += qdot4_exact(wl, x0_thread, s, b, sum0);
          result1[row] += qdot4_exact(wl, x1_thread, s, b, sum1);
          result2[row] += qdot4_exact(wl, x2_thread, s, b, sum2);
        }
      }

      ws += BLOCK_SIZE * BYTES_PER_PACK / PACK_FACTOR;
      sc += BLOCK_SIZE / GS;
      bs += BLOCK_SIZE / GS;
      x0 += BLOCK_SIZE;
      x1 += BLOCK_SIZE;
      x2 += BLOCK_SIZE;
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        float r0 = simd_sum(result0[row]);
        float r1 = simd_sum(result1[row]);
        float r2 = simd_sum(result2[row]);
        if (simd_lid == 0) {
          y[n] = T(r0);
          y[N + n] = T(r1);
          y[2 * N + n] = T(r2);
        }
      }
    }
"#;

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

struct MetalKernelConfig {
    raw: mlx_sys::mlx_fast_metal_kernel_config,
}

impl MetalKernelConfig {
    fn new() -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_fast_metal_kernel_config_new() };
        if raw.ctx.is_null() {
            bail!("failed to create MLX Metal kernel config");
        }
        Ok(Self { raw })
    }

    fn add_output_arg(&self, shape: &[i32], dtype: Dtype) -> Result<()> {
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_add_output_arg(
                self.raw,
                shape.as_ptr(),
                shape.len(),
                dtype as mlx_sys::mlx_dtype,
            )
        };
        status_ok(status, "add Metal output arg")
    }

    fn add_template_arg(&self, arg: &TemplateArg<'_>) -> Result<()> {
        match arg {
            TemplateArg::Dtype(name, dtype) => {
                let name = CString::new(*name)?;
                let status = unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_dtype(
                        self.raw,
                        name.as_ptr(),
                        *dtype as mlx_sys::mlx_dtype,
                    )
                };
                status_ok(status, "add Metal dtype template arg")
            }
            TemplateArg::Int(name, value) => {
                let name = CString::new(*name)?;
                let status = unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_int(
                        self.raw,
                        name.as_ptr(),
                        *value,
                    )
                };
                status_ok(status, "add Metal int template arg")
            }
            TemplateArg::Bool(name, value) => {
                let name = CString::new(*name)?;
                let status = unsafe {
                    mlx_sys::mlx_fast_metal_kernel_config_add_template_arg_bool(
                        self.raw,
                        name.as_ptr(),
                        *value,
                    )
                };
                status_ok(status, "add Metal bool template arg")
            }
        }
    }

    fn set_grid(&self, grid: (i32, i32, i32)) -> Result<()> {
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_grid(self.raw, grid.0, grid.1, grid.2)
        };
        status_ok(status, "set Metal grid")
    }

    fn set_thread_group(&self, group: (i32, i32, i32)) -> Result<()> {
        let status = unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_set_thread_group(
                self.raw, group.0, group.1, group.2,
            )
        };
        status_ok(status, "set Metal thread group")
    }
}

impl Drop for MetalKernelConfig {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_fast_metal_kernel_config_free(self.raw);
        }
    }
}

struct CStringVec {
    raw: mlx_sys::mlx_vector_string,
    _items: Vec<CString>,
}

impl CStringVec {
    fn new(values: &[&str]) -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_vector_string_new() };
        let mut items = Vec::with_capacity(values.len());
        for value in values {
            let item = CString::new(*value)?;
            let status = unsafe {
                mlx_sys::mlx_vector_string_append_value(raw, item.as_ptr() as *const c_char)
            };
            if status != SUCCESS {
                unsafe {
                    mlx_sys::mlx_vector_string_free(raw);
                }
                bail!("failed to append MLX vector string");
            }
            items.push(item);
        }
        Ok(Self { raw, _items: items })
    }

    fn raw(&self) -> mlx_sys::mlx_vector_string {
        self.raw
    }
}

impl Drop for CStringVec {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_vector_string_free(self.raw);
        }
    }
}

struct ArrayVec {
    raw: mlx_sys::mlx_vector_array,
}

impl ArrayVec {
    fn new(values: &[&Array]) -> Result<Self> {
        let raw = unsafe { mlx_sys::mlx_vector_array_new() };
        for value in values {
            let status = unsafe { mlx_sys::mlx_vector_array_append_value(raw, value.as_ptr()) };
            if status != SUCCESS {
                unsafe {
                    mlx_sys::mlx_vector_array_free(raw);
                }
                bail!("failed to append MLX array vector");
            }
        }
        Ok(Self { raw })
    }

    fn into_arrays(self) -> Result<Vec<Array>> {
        let size = unsafe { mlx_sys::mlx_vector_array_size(self.raw) };
        let mut arrays = Vec::with_capacity(size);
        for index in 0..size {
            let mut raw_array = unsafe { mlx_sys::mlx_array_new() };
            let status =
                unsafe { mlx_sys::mlx_vector_array_get(&mut raw_array as *mut _, self.raw, index) };
            if status != SUCCESS {
                unsafe {
                    mlx_sys::mlx_array_free(raw_array);
                }
                return Err(anyhow!("failed to read MLX output array {index}"));
            }
            arrays.push(unsafe { Array::from_ptr(raw_array) });
        }
        Ok(arrays)
    }
}

impl Drop for ArrayVec {
    fn drop(&mut self) {
        unsafe {
            mlx_sys::mlx_vector_array_free(self.raw);
        }
    }
}

fn status_ok(status: i32, action: &str) -> Result<()> {
    if status == SUCCESS {
        Ok(())
    } else {
        Err(anyhow!("{action} failed with status {status}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn custom_metal_kernel_adds_one() -> Result<()> {
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let input = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]);
        let kernel = MetalKernel::new(
            "mtplx_test_add_one",
            &["x"],
            &["out"],
            "uint idx = thread_position_in_grid.x; out[idx] = x[idx] + 1.0f;",
            "",
        )?;
        let outputs = kernel.apply(
            &[&input],
            &[OutputSpec {
                shape: &[4],
                dtype: Dtype::Float32,
            }],
            &[],
            (4, 1, 1),
            (4, 1, 1),
            &Stream::gpu(),
        )?;
        assert_eq!(outputs.len(), 1);
        outputs[0].eval()?;
        assert_eq!(outputs[0].as_slice::<f32>(), &[2.0, 3.0, 4.0, 5.0]);
        Ok(())
    }
}
