use std::cell::RefCell;
use std::sync::OnceLock;

use anyhow::{Result, anyhow, bail};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, Stream};

use super::{MetalKernel, OutputSpec, TemplateArg, env_flag, env_i32, metal_is_available};

pub fn small_m_qmm4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_SMALL_M_QMM4", false))
}

pub fn small_m_qmv4_enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("FERRITE_SMALL_M_QMV4", true))
}

fn small_m_qmv4_max_m() -> i32 {
    static MAX_M: OnceLock<i32> = OnceLock::new();
    *MAX_M.get_or_init(|| env_i32("FERRITE_SMALL_M_QMV4_MAX_M", 1).clamp(1, 6))
}

fn small_m_qmv4_m_enabled(m: i32) -> bool {
    static VALUES: OnceLock<[bool; 7]> = OnceLock::new();
    let values = VALUES.get_or_init(|| {
        if let Ok(value) = std::env::var("FERRITE_SMALL_M_QMV4_M_VALUES") {
            return small_m_qmv4_m_values_from_str(&value).unwrap_or(DEFAULT_SMALL_M_QMV4_M_VALUES);
        }
        if std::env::var_os("FERRITE_SMALL_M_QMV4_MAX_M").is_some() {
            return small_m_qmv4_m_values_to_max(small_m_qmv4_max_m());
        }
        DEFAULT_SMALL_M_QMV4_M_VALUES
    });
    usize::try_from(m)
        .ok()
        .and_then(|index| values.get(index))
        .copied()
        .unwrap_or(false)
}

const DEFAULT_SMALL_M_QMV4_M_VALUES: [bool; 7] = [false, true, false, false, false, true, true];

fn small_m_qmv4_m_values_to_max(max_m: i32) -> [bool; 7] {
    let mut values = [false; 7];
    for m in 1..=max_m.clamp(1, 6) {
        values[m as usize] = true;
    }
    values
}

pub(super) fn small_m_qmv4_m_values_from_str(value: &str) -> Option<[bool; 7]> {
    let normalized = value.trim().to_ascii_lowercase();
    if matches!(normalized.as_str(), "all" | "true" | "yes" | "on") {
        return Some(small_m_qmv4_m_values_to_max(6));
    }
    if matches!(normalized.as_str(), "none" | "0" | "false" | "no" | "off") {
        return Some([false; 7]);
    }

    let mut values = [false; 7];
    let mut any = false;
    for part in normalized.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let m = part.parse::<usize>().ok()?;
        if !(1..=6).contains(&m) {
            return None;
        }
        values[m] = true;
        any = true;
    }
    any.then_some(values)
}

fn small_m_qmv4_simdgroups() -> i32 {
    static SIMDGROUPS: OnceLock<i32> = OnceLock::new();
    *SIMDGROUPS
        .get_or_init(|| qmv4_simdgroups_or_default(env_i32("FERRITE_SMALL_M_QMV4_SIMDGROUPS", 8)))
}

fn small_m_qmv4_packs_per_thread() -> i32 {
    static PACKS: OnceLock<i32> = OnceLock::new();
    *PACKS.get_or_init(|| {
        qmv4_packs_per_thread_or_default(env_i32("FERRITE_SMALL_M_QMV4_PACKS_PER_THREAD", 2))
    })
}

pub fn small_m_qmv4_strict() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("FERRITE_SMALL_M_QMV4_STRICT", false))
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

pub fn small_m_qmv4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    small_m_qmv4_matmul_impl(
        x,
        linear,
        true,
        small_m_qmv4_simdgroups(),
        small_m_qmv4_packs_per_thread(),
    )
}

pub(crate) fn small_m_qmv4_matmul_for_bench(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
    simdgroups: i32,
    packs_per_thread: i32,
) -> Result<Option<Array>> {
    small_m_qmv4_matmul_impl(
        x,
        linear,
        false,
        qmv4_simdgroups_or_default(simdgroups),
        qmv4_packs_per_thread_or_default(packs_per_thread),
    )
}

fn small_m_qmv4_matmul_impl(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
    require_env_enabled: bool,
    simdgroups: i32,
    packs_per_thread: i32,
) -> Result<Option<Array>> {
    if !small_m_qmv4_is_eligible(x, linear, require_env_enabled, simdgroups, packs_per_thread) {
        return Ok(None);
    }

    let shape = x.shape().to_vec();
    let m = shape[shape.len() - 2];
    let k = shape[shape.len() - 1];
    let n = linear.weight.shape()[0];
    let x2 = x.reshape(&[m, k])?;
    let m_arg = Array::from_int(m);
    let k_arg = Array::from_int(k);
    let n_arg = Array::from_int(n);
    let results_per_simdgroup = 4;
    let bn = results_per_simdgroup * simdgroups;
    let grid_y = simdgroups * ((n + bn - 1) / bn);
    let outputs = with_small_m_qmv4_kernel(x.dtype(), |kernel| {
        kernel.apply(
            &[
                &x2,
                &linear.weight,
                &linear.scales,
                &linear.biases,
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
                TemplateArg::Int("GS", linear.group_size),
                TemplateArg::Int("SG", simdgroups),
                TemplateArg::Int("PPT", packs_per_thread),
            ],
            (32 * m, grid_y, 1),
            (32, simdgroups, 1),
            &Stream::gpu(),
        )
    })?;
    let [mut y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("small-m qmv4 kernel returned wrong output count"))?;
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

fn small_m_qmv4_is_eligible(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
    require_env_enabled: bool,
    simdgroups: i32,
    packs_per_thread: i32,
) -> bool {
    if (require_env_enabled && !small_m_qmv4_enabled()) || !metal_is_available() {
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
    if !(1..=6).contains(&m) {
        return false;
    }
    if require_env_enabled && !small_m_qmv4_m_enabled(m) {
        return false;
    }
    let leading_batch = shape[..shape.len() - 2].iter().copied().product::<i32>();
    if leading_batch != 1 {
        return false;
    }
    let k = shape[shape.len() - 1];
    let values_per_thread = 8 * packs_per_thread;
    let block_size = values_per_thread * 32;
    if !matches!(simdgroups, 1 | 2 | 4 | 8)
        || packs_per_thread != 2
        || linear.group_size < values_per_thread
        || k % block_size != 0
    {
        return false;
    }
    let weight_shape = linear.weight.shape();
    if weight_shape.len() != 2 {
        return false;
    }
    let n = weight_shape[0];
    k == weight_shape[1] * 8 && k % 512 == 0 && n > 0
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
    static SMALL_M_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
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

fn with_small_m_qmv4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => {
            SMALL_M_QMV4_BF16.with(|slot| with_cached_small_m_qmv4_kernel(slot, dtype, f))
        }
        Dtype::Float16 => {
            SMALL_M_QMV4_F16.with(|slot| with_cached_small_m_qmv4_kernel(slot, dtype, f))
        }
        other => bail!("small-m qmv4 does not support dtype {other:?}"),
    }
}

fn with_cached_small_m_qmv4_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "ferrite_small_m_qmv4_bn16_bf16",
            Dtype::Float16 => "ferrite_small_m_qmv4_bn16_f16",
            other => bail!("small-m qmv4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &["x", "w", "scales", "biases", "M_size", "K_size", "N_size"],
            &["y"],
            SMALL_M_QMV4_SOURCE,
            SMALL_M_QMV4_HEADER,
        )?);
    }
    let kernel = slot.borrow();
    f(kernel.as_ref().expect("small-m qmv4 kernel initialized"))
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

fn qmv4_simdgroups_or_default(value: i32) -> i32 {
    if matches!(value, 1 | 2 | 4 | 8) {
        value
    } else {
        4
    }
}

fn qmv4_packs_per_thread_or_default(value: i32) -> i32 {
    if value == 2 { value } else { 2 }
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

const SMALL_M_QMV4_HEADER: &str = r#"
    using namespace metal;

    constant constexpr int SIMD_SIZE = 32;
    constant constexpr int PACK_FACTOR = 8;
    constant constexpr int PACKS_PER_THREAD = 2;
    constant constexpr int VALUES_PER_THREAD = PACK_FACTOR * PACKS_PER_THREAD;
    constant constexpr int BYTES_PER_PACK = 4;
    constant constexpr int BLOCK_SIZE = VALUES_PER_THREAD * SIMD_SIZE;
    constant constexpr int RESULTS_PER_SIMDGROUP = 4;
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

const SMALL_M_QMV4_SOURCE: &str = r#"
    uint m_tile = threadgroup_position_in_grid.x;
    uint n_tile = threadgroup_position_in_grid.y;
    uint simd_gid = simdgroup_index_in_threadgroup;
    uint simd_lid = thread_index_in_simdgroup;

    int K = int(K_size);
    int N = int(N_size);
    constexpr int SCALE_STEP_PER_THREAD = GS / VALUES_PER_THREAD;
    int bn = RESULTS_PER_SIMDGROUP * SG;
    int out_row = int(n_tile) * bn + int(simd_gid) * RESULTS_PER_SIMDGROUP;
    int in_vec_size_w = K * BYTES_PER_PACK / PACK_FACTOR;
    int in_vec_size_g = K / GS;

    const device uint8_t* ws_base =
      (const device uint8_t*)w + out_row * in_vec_size_w
      + int(simd_lid) * PACKS_PER_THREAD * BYTES_PER_PACK;
    const device T* scales_base =
      scales + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* biases_base =
      biases + out_row * in_vec_size_g + int(simd_lid) / SCALE_STEP_PER_THREAD;
    const device T* x_base =
      x + int(m_tile) * K + int(simd_lid) * VALUES_PER_THREAD;

    float result[RESULTS_PER_SIMDGROUP] = {0.0f};
    float x_thread[VALUES_PER_THREAD];

    #pragma clang loop unroll_count(4)
    for (int k_block = 0; k_block < K; k_block += BLOCK_SIZE) {
      const device T* x_block = x_base + k_block;
      float x_sum = load_vector4_exact<T>(x_block, x_thread);

      const device uint8_t* ws_block =
        ws_base + k_block * BYTES_PER_PACK / PACK_FACTOR;
      const device T* scales_block = scales_base + k_block / GS;
      const device T* biases_block = biases_base + k_block / GS;

      for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
        int n = out_row + row;
        if (n < N) {
          const device uint8_t* wl = ws_block + row * in_vec_size_w;
          const device T* sl = scales_block + row * in_vec_size_g;
          const device T* bl = biases_block + row * in_vec_size_g;
          float s = float(sl[0]);
          float b = float(bl[0]);

          result[row] += qdot4_exact(wl, x_thread, s, b, x_sum);
        }
      }
    }

    for (int row = 0; row < RESULTS_PER_SIMDGROUP; ++row) {
      int n = out_row + row;
      if (n < N) {
        float sum = simd_sum(result[row]);
        if (simd_lid == 0) {
          y[int(m_tile) * N + n] = T(sum);
        }
      }
    }
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
