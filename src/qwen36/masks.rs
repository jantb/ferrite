use std::cell::RefCell;
use std::sync::OnceLock;

use anyhow::Result;

pub(super) fn decode_block_causal_mask(
    tokens: i32,
    total_keys: i32,
    prev_len: i32,
    dtype: mlx_rs::Dtype,
) -> Result<mlx_rs::Array> {
    if decode_block_mask_mode() == DecodeBlockMaskMode::Mlx {
        return decode_block_causal_mask_mlx(tokens, total_keys, prev_len, dtype);
    }
    decode_block_causal_mask_cpu(tokens, total_keys, prev_len, dtype)
}

fn decode_block_causal_mask_mlx(
    tokens: i32,
    total_keys: i32,
    prev_len: i32,
    dtype: mlx_rs::Dtype,
) -> Result<mlx_rs::Array> {
    let q_positions = mlx_rs::Array::arange::<i32, i32>(Some(prev_len), prev_len + tokens, None)?
        .reshape(&[tokens, 1])?;
    let key_positions =
        mlx_rs::Array::arange::<i32, i32>(None, total_keys, None)?.reshape(&[1, total_keys])?;
    let allowed = q_positions.ge(&key_positions)?;
    let shape = [tokens, total_keys];
    let zeros = mlx_rs::ops::zeros_dtype(&shape, dtype)?;
    let negative =
        mlx_rs::Array::full::<f32>(&shape, mlx_rs::Array::from_f32(-1.0e9))?.as_dtype(dtype)?;
    Ok(mlx_rs::ops::r#where(allowed, zeros, negative)?.reshape(&[1, 1, tokens, total_keys])?)
}

fn decode_block_causal_mask_cpu(
    tokens: i32,
    total_keys: i32,
    prev_len: i32,
    dtype: mlx_rs::Dtype,
) -> Result<mlx_rs::Array> {
    let key = DecodeBlockMaskKey {
        tokens,
        total_keys,
        prev_len,
        dtype,
    };
    if let Some(mask) = LAST_DECODE_BLOCK_MASK.with(|slot| {
        let slot = slot.borrow();
        slot.as_ref()
            .filter(|entry| entry.key == key)
            .map(|entry| entry.mask.clone())
    }) {
        return Ok(mask);
    }
    let mut values = Vec::with_capacity((tokens * total_keys) as usize);
    for query in 0..tokens {
        let max_key = prev_len + query;
        for key in 0..total_keys {
            values.push(if key <= max_key { 0.0_f32 } else { -1.0e9_f32 });
        }
    }
    let mask = mlx_rs::Array::from_slice(&values, &[1, 1, tokens, total_keys]).as_dtype(dtype)?;
    LAST_DECODE_BLOCK_MASK.with(|slot| {
        *slot.borrow_mut() = Some(DecodeBlockMaskCacheEntry {
            key,
            mask: mask.clone(),
        });
    });
    Ok(mask)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DecodeBlockMaskMode {
    Mlx,
    Cpu,
}

fn decode_block_mask_mode() -> DecodeBlockMaskMode {
    static VALUE: OnceLock<DecodeBlockMaskMode> = OnceLock::new();
    *VALUE.get_or_init(|| {
        match std::env::var("FERRITE_DECODE_BLOCK_MASK")
            .unwrap_or_default()
            .trim()
            .to_ascii_lowercase()
            .as_str()
        {
            "mlx" => DecodeBlockMaskMode::Mlx,
            _ => DecodeBlockMaskMode::Cpu,
        }
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct DecodeBlockMaskKey {
    tokens: i32,
    total_keys: i32,
    prev_len: i32,
    dtype: mlx_rs::Dtype,
}

#[derive(Clone, Debug)]
struct DecodeBlockMaskCacheEntry {
    key: DecodeBlockMaskKey,
    mask: mlx_rs::Array,
}

thread_local! {
    static LAST_DECODE_BLOCK_MASK: RefCell<Option<DecodeBlockMaskCacheEntry>> = const { RefCell::new(None) };
}
