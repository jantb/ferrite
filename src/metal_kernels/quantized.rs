use anyhow::{Result, anyhow};
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::{Array, Dtype, Stream};

mod cache;
mod config;
mod sources;
use super::{OutputSpec, TemplateArg, metal_is_available};
use cache::{
    with_gate_up_swiglu_qmv4_kernel, with_large_m_qmm4_kernel, with_multi3_qmv4_kernel,
    with_small_m_qmm4_kernel, with_small_m_qmv4_kernel, with_xlarge_m_qmm4_kernel,
};
#[cfg(test)]
pub(super) use config::small_m_qmv4_m_values_from_str;
pub use config::{
    gate_up_swiglu_qmv4_enabled, large_m_qmm4_enabled, small_m_qmm4_enabled, small_m_qmv4_enabled,
    small_m_qmv4_strict, tiled_qmm4_enabled, xlarge_m_qmm4_enabled,
};
use config::{
    multi3_qmv4_enabled, qmv4_packs_per_thread_or_default, qmv4_simdgroups_or_default,
    small_m_qmv4_m_enabled, small_m_qmv4_packs_per_thread, small_m_qmv4_simdgroups,
};

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

pub fn tiled_qmm4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !tiled_qmm4_is_eligible(x, linear) {
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
    let m_tiles = m / 8;
    let outputs = with_small_m_qmm4_kernel(x.dtype(), |kernel| {
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
            ],
            (64 * m_tiles, n / 32, 1),
            (64, 1, 1),
            &Stream::gpu(),
        )
    })?;
    let [mut y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("tiled qmm4 kernel returned wrong output count"))?;
    if let Some(bias) = &linear.bias {
        y = y.add(bias)?;
    }
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    }
    Ok(Some(y.reshape(&out_shape)?))
}

pub fn large_m_qmm4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !large_m_qmm4_is_eligible(x, linear) {
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
    let m_tiles = m / 32;
    let outputs = with_large_m_qmm4_kernel(x.dtype(), |kernel| {
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
            ],
            (256 * m_tiles, n / 32, 1),
            (256, 1, 1),
            &Stream::gpu(),
        )
    })?;
    let [mut y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("large-m qmm4 kernel returned wrong output count"))?;
    if let Some(bias) = &linear.bias {
        y = y.add(bias)?;
    }
    let mut out_shape = shape;
    if let Some(last) = out_shape.last_mut() {
        *last = n;
    }
    Ok(Some(y.reshape(&out_shape)?))
}

pub fn xlarge_m_qmm4_matmul(
    x: &Array,
    linear: &crate::mlx_backend::QuantizedLinear,
) -> Result<Option<Array>> {
    if !xlarge_m_qmm4_is_eligible(x, linear) {
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
    let m_tiles = m / 64;
    let outputs = with_xlarge_m_qmm4_kernel(x.dtype(), |kernel| {
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
            ],
            (512 * m_tiles, n / 32, 1),
            (512, 1, 1),
            &Stream::gpu(),
        )
    })?;
    let [mut y]: [Array; 1] = outputs
        .try_into()
        .map_err(|_| anyhow!("xlarge-m qmm4 kernel returned wrong output count"))?;
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

fn tiled_qmm4_is_eligible(x: &Array, linear: &crate::mlx_backend::QuantizedLinear) -> bool {
    if !tiled_qmm4_enabled() || !metal_is_available() {
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
    if !(8..=128).contains(&m) || m % 8 != 0 {
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

fn large_m_qmm4_is_eligible(x: &Array, linear: &crate::mlx_backend::QuantizedLinear) -> bool {
    if !large_m_qmm4_enabled() || !metal_is_available() {
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
    if !(32..=128).contains(&m) || m % 32 != 0 {
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

fn xlarge_m_qmm4_is_eligible(x: &Array, linear: &crate::mlx_backend::QuantizedLinear) -> bool {
    if !xlarge_m_qmm4_enabled() || !metal_is_available() {
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
    if !(64..=128).contains(&m) || m % 64 != 0 {
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
