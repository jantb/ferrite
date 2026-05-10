use anyhow::{Result, bail};

#[derive(Debug)]
pub struct MlpWeights {
    pub gate_proj: crate::mlx_backend::QuantizedLinear,
    pub up_proj: crate::mlx_backend::QuantizedLinear,
    pub down_proj: crate::mlx_backend::QuantizedLinear,
    pub fused_gate_up: Option<crate::mlx_backend::FusedQuantizedLinears>,
}

impl MlpWeights {
    pub fn forward(&self, x: &mlx_rs::Array) -> Result<mlx_rs::Array> {
        if let Some(out) = crate::mlx_backend::compiled_swiglu_mlp_q4(
            x,
            &self.gate_proj,
            &self.up_proj,
            &self.down_proj,
        )? {
            return Ok(out);
        }
        if crate::metal_kernels::gate_up_swiglu_qmv4_enabled() {
            if let Some(fused) = crate::metal_kernels::gate_up_swiglu_qmv4_activation(
                x,
                &self.gate_proj,
                &self.up_proj,
            )? {
                return self.down_proj.forward(&fused);
            }
        }
        let (gate_raw, up) = if let Some(fused) = &self.fused_gate_up {
            let mut parts = fused.forward(x)?;
            if parts.len() != 2 {
                bail!("fused MLP projection returned {} parts", parts.len());
            }
            let up = parts.pop().expect("length checked");
            let gate = parts.pop().expect("length checked");
            (gate, up)
        } else {
            (self.gate_proj.forward(x)?, self.up_proj.forward(x)?)
        };
        let gate = mlx_rs::nn::silu(&gate_raw)?;
        let fused = &gate * &up;
        self.down_proj.forward(&fused)
    }
}
