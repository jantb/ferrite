#![allow(dead_code)]

mod gdn;
mod quantized;

use std::ffi::{CString, c_char};

use anyhow::{Result, anyhow, bail};
use mlx_rs::{Array, Dtype, Stream};

pub use gdn::LinearGdnKernels;
pub(crate) use quantized::small_m_qmv4_matmul_for_bench;
pub use quantized::{
    gate_up_swiglu_qmv4_activation, gate_up_swiglu_qmv4_enabled, large_m_qmm4_enabled,
    large_m_qmm4_matmul, small_m_qmm4_enabled, small_m_qmm4_matmul, small_m_qmv4_enabled,
    small_m_qmv4_matmul, small_m_qmv4_strict, tiled_qmm4_enabled, tiled_qmm4_matmul,
    xlarge_m_qmm4_enabled, xlarge_m_qmm4_matmul,
};

const SUCCESS: i32 = 0;

pub fn metal_is_available() -> bool {
    let mut available = false;
    let status = unsafe { mlx_sys::mlx_metal_is_available(&mut available as *mut _) };
    status == SUCCESS && available
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

fn env_i32(name: &str, default: i32) -> i32 {
    std::env::var(name)
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
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
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
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

    #[test]
    fn linear_norm_gate_matches_reference() -> Result<()> {
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let batch = 1;
        let tokens = 2;
        let hv = 2;
        let dv = 4;
        let y_values = (0..(batch * tokens * hv * dv))
            .map(|idx| (idx as f32 - 5.0) / 7.0)
            .collect::<Vec<_>>();
        let z_values = (0..(batch * tokens * hv * dv))
            .map(|idx| (idx as f32 - 3.0) / 5.0)
            .collect::<Vec<_>>();
        let weight_values = [0.75f32, 1.0, 1.25, 1.5];
        let y = Array::from_slice(&y_values, &[batch, tokens, hv, dv]);
        let z = Array::from_slice(&z_values, &[batch, tokens, hv, dv]);
        let weight = Array::from_slice(&weight_values, &[dv]);

        let kernels = LinearGdnKernels::new()?;
        let out = kernels.norm_gate(&y, &z, &weight, 1.0e-6, batch, tokens, hv, dv)?;
        out.eval()?;

        let mut expected = Vec::with_capacity(y_values.len());
        for row in 0..(batch * tokens * hv) as usize {
            let base = row * dv as usize;
            let rms = (y_values[base..base + dv as usize]
                .iter()
                .map(|value| value * value)
                .sum::<f32>()
                / dv as f32
                + 1.0e-6)
                .sqrt();
            for idx in 0..dv as usize {
                let z = z_values[base + idx];
                let gate = z / (1.0 + (-z).exp());
                expected.push(gate * y_values[base + idx] / rms * weight_values[idx]);
            }
        }

        for (got, expected) in out.as_slice::<f32>().iter().zip(expected) {
            assert!(
                (got - expected).abs() < 1.0e-5,
                "got {got}, expected {expected}"
            );
        }
        Ok(())
    }

    #[test]
    fn small_m_qmv4_matches_quantized_matmul() -> Result<()> {
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let k = 512;
        let n = 24;
        let group_size = 128;
        let dense_w_values = (0..(n * k))
            .map(|idx| (((idx * 17 + 11) % 41) as f32 - 20.0) / 17.0)
            .collect::<Vec<_>>();
        let dense_w = Array::from_slice(&dense_w_values, &[n, k]).as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = mlx_rs::ops::quantize(&dense_w, group_size, 4)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;

        let linear = crate::mlx_backend::QuantizedLinear {
            weight,
            scales,
            biases,
            bias: None,
            group_size,
            bits: 4,
        };

        for m in [1, 3, 6] {
            let x_values = (0..(m * k))
                .map(|idx| (((idx * 13 + 7) % 37) as f32 - 18.0) / 19.0)
                .collect::<Vec<_>>();
            let x = Array::from_slice(&x_values, &[m, k]).as_dtype(Dtype::Bfloat16)?;
            let expected = mlx_rs::ops::quantized_matmul(
                &x,
                &linear.weight,
                &linear.scales,
                &linear.biases,
                true,
                group_size,
                4,
            )?
            .as_dtype(Dtype::Float32)?;
            let actual = small_m_qmv4_matmul_for_bench(&x, &linear, 4, 2)?
                .expect("small-m qmv4 should be eligible")
                .as_dtype(Dtype::Float32)?;
            expected.eval()?;
            actual.eval()?;

            let expected = expected.as_slice::<f32>();
            let actual = actual.as_slice::<f32>();
            assert_eq!(actual.len(), expected.len());
            for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
                let diff = (actual - expected).abs();
                assert!(
                    diff <= 0.125,
                    "m={m} idx={idx} actual={actual} expected={expected} diff={diff}"
                );
            }
        }

        Ok(())
    }

    #[test]
    fn tiled_qmm4_matches_quantized_matmul() -> Result<()> {
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let m = 16;
        let k = 512;
        let n = 64;
        let group_size = 128;
        let dense_w_values = (0..(n * k))
            .map(|idx| (((idx * 19 + 5) % 43) as f32 - 21.0) / 23.0)
            .collect::<Vec<_>>();
        let dense_w = Array::from_slice(&dense_w_values, &[n, k]).as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = mlx_rs::ops::quantize(&dense_w, group_size, 4)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;
        let linear = crate::mlx_backend::QuantizedLinear {
            weight,
            scales,
            biases,
            bias: None,
            group_size,
            bits: 4,
        };
        let x_values = (0..(m * k))
            .map(|idx| (((idx * 13 + 7) % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let x = Array::from_slice(&x_values, &[m, k]).as_dtype(Dtype::Bfloat16)?;
        let expected = mlx_rs::ops::quantized_matmul(
            &x,
            &linear.weight,
            &linear.scales,
            &linear.biases,
            true,
            group_size,
            4,
        )?
        .as_dtype(Dtype::Float32)?;
        let actual = tiled_qmm4_matmul(&x, &linear)?
            .expect("tiled qmm4 should be eligible")
            .as_dtype(Dtype::Float32)?;
        expected.eval()?;
        actual.eval()?;

        let expected = expected.as_slice::<f32>();
        let actual = actual.as_slice::<f32>();
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 0.125,
                "idx={idx} actual={actual} expected={expected} diff={diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn large_m_qmm4_matches_quantized_matmul() -> Result<()> {
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let m = 32;
        let k = 512;
        let n = 64;
        let group_size = 128;
        let dense_w_values = (0..(n * k))
            .map(|idx| (((idx * 19 + 5) % 43) as f32 - 21.0) / 23.0)
            .collect::<Vec<_>>();
        let dense_w = Array::from_slice(&dense_w_values, &[n, k]).as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = mlx_rs::ops::quantize(&dense_w, group_size, 4)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;
        let linear = crate::mlx_backend::QuantizedLinear {
            weight,
            scales,
            biases,
            bias: None,
            group_size,
            bits: 4,
        };
        let x_values = (0..(m * k))
            .map(|idx| (((idx * 13 + 7) % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let x = Array::from_slice(&x_values, &[m, k]).as_dtype(Dtype::Bfloat16)?;
        let expected = mlx_rs::ops::quantized_matmul(
            &x,
            &linear.weight,
            &linear.scales,
            &linear.biases,
            true,
            group_size,
            4,
        )?
        .as_dtype(Dtype::Float32)?;
        let actual = large_m_qmm4_matmul(&x, &linear)?
            .expect("large-m qmm4 should be eligible")
            .as_dtype(Dtype::Float32)?;
        expected.eval()?;
        actual.eval()?;

        let expected = expected.as_slice::<f32>();
        let actual = actual.as_slice::<f32>();
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 0.125,
                "idx={idx} actual={actual} expected={expected} diff={diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn xlarge_m_qmm4_matches_quantized_matmul() -> Result<()> {
        if std::env::var_os("FERRITE_RUN_METAL_TESTS").is_none() {
            return Ok(());
        }
        let _guard = crate::mlx_test_lock();
        if std::env::var_os("MAKELEVEL").is_some() {
            return Ok(());
        }
        if !metal_is_available() {
            return Ok(());
        }

        let m = 64;
        let k = 512;
        let n = 64;
        let group_size = 128;
        let dense_w_values = (0..(n * k))
            .map(|idx| (((idx * 19 + 5) % 43) as f32 - 21.0) / 23.0)
            .collect::<Vec<_>>();
        let dense_w = Array::from_slice(&dense_w_values, &[n, k]).as_dtype(Dtype::Bfloat16)?;
        let (weight, scales, biases) = mlx_rs::ops::quantize(&dense_w, group_size, 4)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;
        let linear = crate::mlx_backend::QuantizedLinear {
            weight,
            scales,
            biases,
            bias: None,
            group_size,
            bits: 4,
        };
        let x_values = (0..(m * k))
            .map(|idx| (((idx * 13 + 7) % 37) as f32 - 18.0) / 19.0)
            .collect::<Vec<_>>();
        let x = Array::from_slice(&x_values, &[m, k]).as_dtype(Dtype::Bfloat16)?;
        let expected = mlx_rs::ops::quantized_matmul(
            &x,
            &linear.weight,
            &linear.scales,
            &linear.biases,
            true,
            group_size,
            4,
        )?
        .as_dtype(Dtype::Float32)?;
        let actual = xlarge_m_qmm4_matmul(&x, &linear)?
            .expect("xlarge-m qmm4 should be eligible")
            .as_dtype(Dtype::Float32)?;
        expected.eval()?;
        actual.eval()?;

        let expected = expected.as_slice::<f32>();
        let actual = actual.as_slice::<f32>();
        assert_eq!(actual.len(), expected.len());
        for (idx, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
            let diff = (actual - expected).abs();
            assert!(
                diff <= 0.125,
                "idx={idx} actual={actual} expected={expected} diff={diff}"
            );
        }
        Ok(())
    }

    #[test]
    fn small_m_qmv4_m_value_selector_parses_lists() {
        let selected = quantized::small_m_qmv4_m_values_from_str("1,5,6").unwrap();
        assert!(selected[1]);
        assert!(!selected[2]);
        assert!(!selected[3]);
        assert!(!selected[4]);
        assert!(selected[5]);
        assert!(selected[6]);

        let all = quantized::small_m_qmv4_m_values_from_str("all").unwrap();
        assert!((1..=6).all(|m| all[m]));

        let none = quantized::small_m_qmv4_m_values_from_str("off").unwrap();
        assert!((1..=6).all(|m| !none[m]));

        assert!(quantized::small_m_qmv4_m_values_from_str("7").is_none());
    }
}
