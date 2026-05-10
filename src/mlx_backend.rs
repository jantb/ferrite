#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use mlx_rs::Array;
use mlx_rs::fast::rms_norm;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::{concatenate_axis, dequantize, quantized_matmul};
use mlx_rs::transforms::compile::compile;

#[derive(Debug)]
pub struct MlxWeightStore {
    pub arrays: HashMap<String, Array>,
    pub source_files: BTreeSet<PathBuf>,
}

impl MlxWeightStore {
    pub fn load_model_dir(
        model_dir: &Path,
        tensors: &crate::artifacts::ModelTensors,
    ) -> Result<Self> {
        let mut arrays = HashMap::new();
        let mut source_files = BTreeSet::new();

        let mut by_file: BTreeMap<PathBuf, Vec<String>> = BTreeMap::new();
        for (name, info) in tensors
            .model_tensors
            .iter()
            .chain(tensors.mtp_tensors.iter())
        {
            by_file
                .entry(info.source_file.clone())
                .or_default()
                .push(name.clone());
        }

        for (file, wanted_names) in by_file {
            let loaded = Array::load_safetensors(&file)
                .with_context(|| format!("load MLX safetensors {}", file.display()))?;
            source_files.insert(file.clone());
            for name in wanted_names {
                if let Some(array) = loaded.get(&name) {
                    arrays.insert(name, array.clone());
                }
            }
        }

        let _ = model_dir;
        Ok(Self {
            arrays,
            source_files,
        })
    }

    pub fn len(&self) -> usize {
        self.arrays.len()
    }

    pub fn quantized_linear(
        &self,
        prefix: impl AsRef<str>,
        group_size: i32,
        bits: i32,
    ) -> Result<QuantizedLinear> {
        let prefix = prefix.as_ref();
        Ok(QuantizedLinear {
            weight: self.array(&format!("{prefix}.weight"))?.clone(),
            scales: self.array(&format!("{prefix}.scales"))?.clone(),
            biases: self.array(&format!("{prefix}.biases"))?.clone(),
            bias: self.arrays.get(&format!("{prefix}.bias")).cloned(),
            group_size,
            bits,
        })
    }

    pub fn array(&self, name: &str) -> Result<&Array> {
        self.arrays
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("missing MLX tensor {name}"))
    }
}

#[derive(Clone, Debug)]
pub struct QuantizedLinear {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub bias: Option<Array>,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedLinear {
    pub fn forward(&self, x: &Array) -> Result<Array> {
        if crate::metal_kernels::small_m_qmv4_enabled() {
            match crate::metal_kernels::small_m_qmv4_matmul(x, self) {
                Ok(Some(y)) => return Ok(y),
                Ok(None) => {}
                Err(err) => {
                    if crate::metal_kernels::small_m_qmv4_strict() {
                        return Err(err).context("small-m qmv4 fast path failed");
                    }
                }
            }
        }
        if crate::metal_kernels::large_m_qmm4_enabled() {
            if let Some(y) = crate::metal_kernels::large_m_qmm4_matmul(x, self)? {
                return Ok(y);
            }
        }
        if crate::metal_kernels::xlarge_m_qmm4_enabled() {
            if let Some(y) = crate::metal_kernels::xlarge_m_qmm4_matmul(x, self)? {
                return Ok(y);
            }
        }
        if crate::metal_kernels::tiled_qmm4_enabled() {
            if let Some(y) = crate::metal_kernels::tiled_qmm4_matmul(x, self)? {
                return Ok(y);
            }
        }
        if crate::metal_kernels::small_m_qmm4_enabled() {
            if let Some(y) = crate::metal_kernels::small_m_qmm4_matmul(x, self)? {
                return Ok(y);
            }
        }
        let mut y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        if let Some(bias) = &self.bias {
            y = y.add(bias)?;
        }
        Ok(y)
    }

    fn compatible_for_fusion(&self, other: &Self) -> bool {
        self.bias.is_none()
            && other.bias.is_none()
            && self.group_size == other.group_size
            && self.bits == other.bits
            && trailing_shape(self.weight.shape()) == trailing_shape(other.weight.shape())
            && trailing_shape(self.scales.shape()) == trailing_shape(other.scales.shape())
            && trailing_shape(self.biases.shape()) == trailing_shape(other.biases.shape())
    }

    pub fn requantize(&self, group_size: i32, bits: i32) -> Result<Self> {
        let dequantized = dequantize(
            &self.weight,
            &self.scales,
            &self.biases,
            self.group_size,
            self.bits,
        )?;
        let (weight, scales, biases) = mlx_rs::ops::quantize(&dequantized, group_size, bits)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;
        Ok(Self {
            weight,
            scales,
            biases,
            bias: self.bias.clone(),
            group_size,
            bits,
        })
    }
}

pub fn compiled_swiglu_mlp_q4(
    x: &Array,
    gate: &QuantizedLinear,
    up: &QuantizedLinear,
    down: &QuantizedLinear,
) -> Result<Option<Array>> {
    if !compiled_swiglu_mlp_q4_is_eligible(x, gate, up, down) {
        return Ok(None);
    }

    let args = [
        x.clone(),
        gate.weight.clone(),
        gate.scales.clone(),
        gate.biases.clone(),
        up.weight.clone(),
        up.scales.clone(),
        up.biases.clone(),
        down.weight.clone(),
        down.scales.clone(),
        down.biases.clone(),
    ];
    let mut outputs = match gate.group_size {
        32 => {
            let mut compiled = compile(compiled_swiglu_mlp_q4_gs32, true);
            compiled(&args)?
        }
        64 => {
            let mut compiled = compile(compiled_swiglu_mlp_q4_gs64, true);
            compiled(&args)?
        }
        128 => {
            let mut compiled = compile(compiled_swiglu_mlp_q4_gs128, true);
            compiled(&args)?
        }
        _ => return Ok(None),
    };
    Ok(outputs.pop())
}

fn compiled_swiglu_mlp_q4_is_eligible(
    x: &Array,
    gate: &QuantizedLinear,
    up: &QuantizedLinear,
    down: &QuantizedLinear,
) -> bool {
    if !compiled_swiglu_mlp_q4_enabled() {
        return false;
    }
    if gate.bits != 4 || up.bits != 4 || down.bits != 4 {
        return false;
    }
    if gate.group_size != up.group_size || gate.group_size != down.group_size {
        return false;
    }
    if !matches!(gate.group_size, 32 | 64 | 128) {
        return false;
    }
    if gate.bias.is_some() || up.bias.is_some() || down.bias.is_some() {
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
    if gate.weight.shape() != up.weight.shape()
        || gate.scales.shape() != up.scales.shape()
        || gate.biases.shape() != up.biases.shape()
    {
        return false;
    }
    let k = shape[shape.len() - 1];
    let gate_shape = gate.weight.shape();
    let down_shape = down.weight.shape();
    gate_shape.len() == 2
        && down_shape.len() == 2
        && k == gate_shape[1] * 8
        && gate_shape[0] == down_shape[1] * 8
}

fn compiled_swiglu_mlp_q4_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| env_flag("MTPLX_COMPILED_MLP_Q4", false))
}

fn compiled_swiglu_mlp_q4_gs32(
    args: &[Array],
) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    compiled_swiglu_mlp_q4_impl(args, 32)
}

fn compiled_swiglu_mlp_q4_gs64(
    args: &[Array],
) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    compiled_swiglu_mlp_q4_impl(args, 64)
}

fn compiled_swiglu_mlp_q4_gs128(
    args: &[Array],
) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    compiled_swiglu_mlp_q4_impl(args, 128)
}

fn compiled_swiglu_mlp_q4_impl(
    args: &[Array],
    group_size: i32,
) -> std::result::Result<Vec<Array>, mlx_rs::error::Exception> {
    let gate = quantized_matmul(&args[0], &args[1], &args[2], &args[3], true, group_size, 4)?;
    let up = quantized_matmul(&args[0], &args[4], &args[5], &args[6], true, group_size, 4)?;
    let gate = mlx_rs::nn::silu(&gate)?;
    let fused = &gate * &up;
    let out = quantized_matmul(&fused, &args[7], &args[8], &args[9], true, group_size, 4)?;
    Ok(vec![out])
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

#[derive(Clone, Debug)]
pub struct Linear {
    pub weight: Array,
    pub bias: Option<Array>,
}

impl Linear {
    pub fn from_store(store: &MlxWeightStore, prefix: impl AsRef<str>) -> Result<Self> {
        let prefix = prefix.as_ref();
        Ok(Self {
            weight: store.array(&format!("{prefix}.weight"))?.clone(),
            bias: store.arrays.get(&format!("{prefix}.bias")).cloned(),
        })
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        let mut y = x.matmul(self.weight.t())?;
        if let Some(bias) = &self.bias {
            y = y.add(bias)?;
        }
        Ok(y)
    }
}

#[derive(Debug)]
pub struct FusedQuantizedLinears {
    weight: Array,
    scales: Array,
    biases: Array,
    split_points: Vec<i32>,
    group_size: i32,
    bits: i32,
}

impl FusedQuantizedLinears {
    pub fn try_new(linears: &[&QuantizedLinear]) -> Result<Option<Self>> {
        let Some(first) = linears.first().copied() else {
            return Ok(None);
        };
        if linears
            .iter()
            .skip(1)
            .any(|linear| !first.compatible_for_fusion(linear))
        {
            return Ok(None);
        }
        if first.bias.is_some() {
            return Ok(None);
        }

        let mut split_points = Vec::with_capacity(linears.len().saturating_sub(1));
        let mut running = 0;
        for linear in linears.iter().take(linears.len().saturating_sub(1)) {
            let Some(width) = linear.weight.shape().first().copied() else {
                bail!("quantized linear weight is scalar");
            };
            running += width;
            split_points.push(running);
        }

        let weights = linears
            .iter()
            .map(|linear| linear.weight.clone())
            .collect::<Vec<_>>();
        let scales = linears
            .iter()
            .map(|linear| linear.scales.clone())
            .collect::<Vec<_>>();
        let biases = linears
            .iter()
            .map(|linear| linear.biases.clone())
            .collect::<Vec<_>>();
        let weight = concatenate_axis(&weights, 0)?;
        let scales = concatenate_axis(&scales, 0)?;
        let biases = concatenate_axis(&biases, 0)?;
        weight.eval()?;
        scales.eval()?;
        biases.eval()?;

        Ok(Some(Self {
            weight,
            scales,
            biases,
            split_points,
            group_size: first.group_size,
            bits: first.bits,
        }))
    }

    pub fn forward(&self, x: &Array) -> Result<Vec<Array>> {
        let y = quantized_matmul(
            x,
            &self.weight,
            &self.scales,
            &self.biases,
            true,
            self.group_size,
            self.bits,
        )?;
        Ok(y.split_axis(&self.split_points, -1)?)
    }
}

fn trailing_shape(shape: &[i32]) -> &[i32] {
    shape.get(1..).unwrap_or(&[])
}

#[derive(Clone, Debug)]
pub struct QuantizedEmbedding {
    pub weight: Array,
    pub scales: Array,
    pub biases: Array,
    pub group_size: i32,
    pub bits: i32,
}

impl QuantizedEmbedding {
    pub fn from_store(
        store: &MlxWeightStore,
        prefix: impl AsRef<str>,
        group_size: i32,
        bits: i32,
    ) -> Result<Self> {
        let prefix = prefix.as_ref();
        Ok(Self {
            weight: store.array(&format!("{prefix}.weight"))?.clone(),
            scales: store.array(&format!("{prefix}.scales"))?.clone(),
            biases: store.array(&format!("{prefix}.biases"))?.clone(),
            group_size,
            bits,
        })
    }

    pub fn forward(&self, input_ids: &Array) -> Result<Array> {
        let shape = input_ids.shape();
        let flat = input_ids.flatten(None, None)?;
        let w = self.weight.index(&flat);
        let scales = self.scales.index(&flat);
        let biases = self.biases.index(&flat);
        let out = dequantize(&w, &scales, &biases, self.group_size, self.bits)?;
        let ret_shape = shape
            .iter()
            .copied()
            .chain(std::iter::once(-1))
            .collect::<Vec<_>>();
        Ok(out.reshape(&ret_shape)?)
    }
}

#[derive(Clone, Debug)]
pub struct RmsNorm {
    pub weight: Array,
    pub eps: f32,
}

impl RmsNorm {
    pub fn new(weight: Array, eps: f32) -> Self {
        Self { weight, eps }
    }

    pub fn forward(&self, x: &Array) -> Result<Array> {
        Ok(rms_norm(x, &self.weight, self.eps)?)
    }
}
