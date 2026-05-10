use std::time::Instant;

use super::DecodeProfileTimings;
use anyhow::{Result, bail};

#[derive(Debug)]
pub struct MlpWeights {
    gate_up: GateUpProjections,
    pub down_proj: crate::mlx_backend::QuantizedLinear,
}

#[derive(Debug)]
enum GateUpProjections {
    Separate {
        gate_proj: crate::mlx_backend::QuantizedLinear,
        up_proj: crate::mlx_backend::QuantizedLinear,
    },
    Fused(crate::mlx_backend::FusedQuantizedLinears),
}

impl MlpWeights {
    pub fn new(
        gate_proj: crate::mlx_backend::QuantizedLinear,
        up_proj: crate::mlx_backend::QuantizedLinear,
        down_proj: crate::mlx_backend::QuantizedLinear,
        fuse_gate_up: bool,
    ) -> Result<Self> {
        if fuse_gate_up {
            if let Some(fused) =
                crate::mlx_backend::FusedQuantizedLinears::try_new(&[&gate_proj, &up_proj])?
            {
                return Ok(Self {
                    gate_up: GateUpProjections::Fused(fused),
                    down_proj,
                });
            }
        }
        Ok(Self {
            gate_up: GateUpProjections::Separate { gate_proj, up_proj },
            down_proj,
        })
    }

    pub fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        if let GateUpProjections::Separate { gate_proj, up_proj } = &self.gate_up {
            if let Some(out) =
                crate::mlx_backend::compiled_swiglu_mlp_q4(x, gate_proj, up_proj, &self.down_proj)?
            {
                return Ok(out);
            }
            if crate::metal_kernels::gate_up_swiglu_qmv4_enabled() {
                if let Some(fused) =
                    crate::metal_kernels::gate_up_swiglu_qmv4_activation(x, gate_proj, up_proj)?
                {
                    return self.down_proj.forward(&fused);
                }
            }
        }
        let (gate_raw, up) = match &self.gate_up {
            GateUpProjections::Fused(fused) => {
                let mut parts = fused.forward(x)?;
                if parts.len() != 2 {
                    bail!("fused MLP projection returned {} parts", parts.len());
                }
                let up = parts.pop().expect("length checked");
                let gate = parts.pop().expect("length checked");
                (gate, up)
            }
            GateUpProjections::Separate { gate_proj, up_proj } => {
                (gate_proj.forward(x)?, up_proj.forward(x)?)
            }
        };
        let gate = mlx_rs::nn::silu(&gate_raw)?;
        let fused = &gate * &up;
        self.down_proj.forward(&fused)
    }

    pub fn forward_profiled(
        &self,
        x: &mlx_rs::Array,
        profile: &mut DecodeProfileTimings,
    ) -> Result<mlx_rs::Array> {
        if let GateUpProjections::Separate { gate_proj, up_proj } = &self.gate_up {
            let started = Instant::now();
            if let Some(out) =
                crate::mlx_backend::compiled_swiglu_mlp_q4(x, gate_proj, up_proj, &self.down_proj)?
            {
                out.eval()?;
                profile.mlp_compiled_s += started.elapsed().as_secs_f64();
                return Ok(out);
            }

            if crate::metal_kernels::gate_up_swiglu_qmv4_enabled() {
                let started = Instant::now();
                if let Some(fused) =
                    crate::metal_kernels::gate_up_swiglu_qmv4_activation(x, gate_proj, up_proj)?
                {
                    fused.eval()?;
                    profile.mlp_gate_up_s += started.elapsed().as_secs_f64();

                    let started = Instant::now();
                    let out = self.down_proj.forward(&fused)?;
                    out.eval()?;
                    profile.mlp_down_s += started.elapsed().as_secs_f64();
                    return Ok(out);
                }
            }
        }

        let started = Instant::now();
        let (gate_raw, up) = match &self.gate_up {
            GateUpProjections::Fused(fused) => {
                let mut parts = fused.forward(x)?;
                if parts.len() != 2 {
                    bail!("fused MLP projection returned {} parts", parts.len());
                }
                let up = parts.pop().expect("length checked");
                let gate = parts.pop().expect("length checked");
                (gate, up)
            }
            GateUpProjections::Separate { gate_proj, up_proj } => {
                (gate_proj.forward(x)?, up_proj.forward(x)?)
            }
        };
        gate_raw.eval()?;
        up.eval()?;
        profile.mlp_gate_up_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let gate = mlx_rs::nn::silu(&gate_raw)?;
        let fused = &gate * &up;
        fused.eval()?;
        profile.mlp_activation_s += started.elapsed().as_secs_f64();

        let started = Instant::now();
        let out = self.down_proj.forward(&fused)?;
        out.eval()?;
        profile.mlp_down_s += started.elapsed().as_secs_f64();
        Ok(out)
    }
}
