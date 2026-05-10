use std::cell::RefCell;

use anyhow::{Result, bail};
use mlx_rs::Dtype;

use super::super::MetalKernel;
use super::sources::{
    GATE_UP_SWIGLU_QMV4_HEADER, GATE_UP_SWIGLU_QMV4_SOURCE, LARGE_M_QMM4_SOURCE,
    MULTI3_QMV4_HEADER, MULTI3_QMV4_SOURCE, SMALL_M_QMM4_SOURCE, SMALL_M_QMV4_HEADER,
    SMALL_M_QMV4_SOURCE, XLARGE_M_QMM4_SOURCE,
};

thread_local! {
    static GATE_UP_SWIGLU_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static GATE_UP_SWIGLU_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMM4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMM4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static LARGE_M_QMM4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static LARGE_M_QMM4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static XLARGE_M_QMM4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static XLARGE_M_QMM4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static SMALL_M_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static MULTI3_QMV4_BF16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
    static MULTI3_QMV4_F16: RefCell<Option<MetalKernel>> = const { RefCell::new(None) };
}

pub(super) fn with_gate_up_swiglu_qmv4_kernel<T>(
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

pub(super) fn with_small_m_qmm4_kernel<T>(
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

pub(super) fn with_large_m_qmm4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => {
            LARGE_M_QMM4_BF16.with(|slot| with_cached_large_m_kernel(slot, dtype, f))
        }
        Dtype::Float16 => LARGE_M_QMM4_F16.with(|slot| with_cached_large_m_kernel(slot, dtype, f)),
        other => bail!("large-m qmm4 does not support dtype {other:?}"),
    }
}

pub(super) fn with_xlarge_m_qmm4_kernel<T>(
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    match dtype {
        Dtype::Bfloat16 => {
            XLARGE_M_QMM4_BF16.with(|slot| with_cached_xlarge_m_kernel(slot, dtype, f))
        }
        Dtype::Float16 => {
            XLARGE_M_QMM4_F16.with(|slot| with_cached_xlarge_m_kernel(slot, dtype, f))
        }
        other => bail!("xlarge-m qmm4 does not support dtype {other:?}"),
    }
}

fn with_cached_xlarge_m_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "mtplx_rs_xlarge_m_qmm4_bf16",
            Dtype::Float16 => "mtplx_rs_xlarge_m_qmm4_f16",
            other => bail!("xlarge-m qmm4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
            &["y"],
            XLARGE_M_QMM4_SOURCE,
            "",
        )?);
    }
    let kernel = slot.borrow();
    f(kernel.as_ref().expect("xlarge-m qmm4 kernel initialized"))
}

fn with_cached_large_m_kernel<T>(
    slot: &RefCell<Option<MetalKernel>>,
    dtype: Dtype,
    f: impl FnOnce(&MetalKernel) -> Result<T>,
) -> Result<T> {
    if slot.borrow().is_none() {
        let name = match dtype {
            Dtype::Bfloat16 => "mtplx_rs_large_m_qmm4_bf16",
            Dtype::Float16 => "mtplx_rs_large_m_qmm4_f16",
            other => bail!("large-m qmm4 does not support dtype {other:?}"),
        };
        *slot.borrow_mut() = Some(MetalKernel::new(
            name,
            &["x", "w_q", "scales", "biases", "M_size", "K_size", "N_size"],
            &["y"],
            LARGE_M_QMM4_SOURCE,
            "",
        )?);
    }
    let kernel = slot.borrow();
    f(kernel.as_ref().expect("large-m qmm4 kernel initialized"))
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

pub(super) fn with_small_m_qmv4_kernel<T>(
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

pub(super) fn with_multi3_qmv4_kernel<T>(
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
