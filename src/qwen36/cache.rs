use anyhow::{Result, bail};
use mlx_rs::ops::concatenate_axis;
use mlx_rs::ops::indexing::IndexOp;
use mlx_rs::ops::indexing::TryIndexMutOp;

#[derive(Clone, Debug)]
pub struct DecodeState {
    pub position: i32,
    pub layers: Vec<LayerDecodeState>,
}

#[derive(Clone, Debug)]
pub enum LayerDecodeState {
    Full(FullAttentionCache),
    Linear(LinearAttentionCache),
}

#[derive(Clone, Debug, Default)]
pub struct FullAttentionCache {
    pub k: Option<mlx_rs::Array>,
    pub v: Option<mlx_rs::Array>,
    pub offset: i32,
}

impl FullAttentionCache {
    fn capacity_step() -> i32 {
        std::env::var("FERRITE_FULL_KV_CACHE_STEP")
            .ok()
            .and_then(|value| value.trim().parse::<i32>().ok())
            .filter(|value| *value > 0)
            .unwrap_or(256)
    }

    pub(super) fn current_len(&self) -> i32 {
        self.offset.max(0)
    }

    fn capacity(&self) -> i32 {
        self.k
            .as_ref()
            .and_then(|array| array.shape().get(2).copied())
            .unwrap_or(0)
    }

    pub fn truncate(&mut self, new_len: i32) {
        self.offset = new_len.clamp(0, self.capacity());
    }

    pub fn set_prefill(&mut self, k: mlx_rs::Array, v: mlx_rs::Array) -> Result<()> {
        let k_shape = k.shape();
        let v_shape = v.shape();
        if k_shape.len() != 4 || v_shape.len() != 4 {
            bail!("full-attention prefill cache expects [batch, heads, tokens, dim]");
        }
        if k_shape[0..3] != v_shape[0..3] {
            bail!(
                "full-attention prefill key/value cache shapes do not match: {k_shape:?} vs {v_shape:?}"
            );
        }
        let len = k_shape[2];
        if full_kv_cache_append_mode() == FullKvCacheAppendMode::Concat
            || !prefill_reserve_enabled()
        {
            self.offset = len;
            self.k = Some(k);
            self.v = Some(v);
            return Ok(());
        }

        self.offset = 0;
        self.k = None;
        self.v = None;
        self.ensure_capacity(len + 1, &k, &v)?;
        self.k
            .as_mut()
            .expect("full-attention key cache was just allocated")
            .try_index_mut((.., .., 0..len, ..), k)?;
        self.v
            .as_mut()
            .expect("full-attention value cache was just allocated")
            .try_index_mut((.., .., 0..len, ..), v)?;
        self.offset = len;
        Ok(())
    }

    pub fn active_k(&self) -> Result<mlx_rs::Array> {
        let k = self
            .k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing full-attention key cache"))?;
        let len = self.current_len();
        if len >= k.shape()[2] {
            Ok(k.clone())
        } else {
            Ok(k.index((.., .., 0..len, ..)))
        }
    }

    pub fn active_v(&self) -> Result<mlx_rs::Array> {
        let v = self
            .v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing full-attention value cache"))?;
        let len = self.current_len();
        if len >= v.shape()[2] {
            Ok(v.clone())
        } else {
            Ok(v.index((.., .., 0..len, ..)))
        }
    }

    pub fn append_and_fetch(
        &mut self,
        k: mlx_rs::Array,
        v: mlx_rs::Array,
    ) -> Result<(mlx_rs::Array, mlx_rs::Array, i32)> {
        let prev = self.append_without_fetch(k, v)?;
        Ok((self.active_k()?, self.active_v()?, prev))
    }

    pub fn append_without_fetch(&mut self, k: mlx_rs::Array, v: mlx_rs::Array) -> Result<i32> {
        let k_shape = k.shape();
        let v_shape = v.shape();
        if k_shape.len() != 4 || v_shape.len() != 4 {
            bail!("full-attention cache append expects [batch, heads, tokens, dim]");
        }
        if k_shape[0..3] != v_shape[0..3] {
            bail!("full-attention key/value cache shapes do not match: {k_shape:?} vs {v_shape:?}");
        }
        let prev = self.current_len();
        let steps = k_shape[2];
        let end = prev + steps;
        if full_kv_cache_append_mode() == FullKvCacheAppendMode::Concat {
            let next_k = match self.k.take() {
                Some(prev_k) if prev > 0 => {
                    let active = if prev >= prev_k.shape()[2] {
                        prev_k
                    } else {
                        prev_k.index((.., .., 0..prev, ..))
                    };
                    concatenate_axis(&[active, k], 2)?
                }
                _ => k,
            };
            let next_v = match self.v.take() {
                Some(prev_v) if prev > 0 => {
                    let active = if prev >= prev_v.shape()[2] {
                        prev_v
                    } else {
                        prev_v.index((.., .., 0..prev, ..))
                    };
                    concatenate_axis(&[active, v], 2)?
                }
                _ => v,
            };
            self.offset = end;
            self.k = Some(next_k);
            self.v = Some(next_v);
            return Ok(prev);
        }
        self.ensure_capacity(end, &k, &v)?;
        self.k
            .as_mut()
            .expect("full-attention key cache was just allocated")
            .try_index_mut((.., .., prev..end, ..), k)?;
        self.v
            .as_mut()
            .expect("full-attention value cache was just allocated")
            .try_index_mut((.., .., prev..end, ..), v)?;
        self.offset = end;
        Ok(prev)
    }

    pub fn active_block_slices(
        &self,
        block_tokens: i32,
    ) -> Result<Vec<(i32, mlx_rs::Array, mlx_rs::Array)>> {
        if block_tokens <= 0 {
            bail!("full-attention block size must be positive, got {block_tokens}");
        }
        let k = self
            .k
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing full-attention key cache"))?;
        let v = self
            .v
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("missing full-attention value cache"))?;
        let len = self.current_len();
        let mut blocks = Vec::new();
        let mut start = 0;
        while start < len {
            let end = (start + block_tokens).min(len);
            blocks.push((
                start,
                k.index((.., .., start..end, ..)),
                v.index((.., .., start..end, ..)),
            ));
            start = end;
        }
        Ok(blocks)
    }

    fn ensure_capacity(
        &mut self,
        required: i32,
        k: &mlx_rs::Array,
        v: &mlx_rs::Array,
    ) -> Result<()> {
        if self.k.is_some() && self.v.is_some() && self.capacity() >= required {
            return Ok(());
        }

        let prev = self.current_len();
        let step = Self::capacity_step();
        let capacity = ((required + step - 1) / step) * step;
        let k_shape = k.shape();
        let v_shape = v.shape();
        let new_k_shape = [k_shape[0], k_shape[1], capacity, k_shape[3]];
        let new_v_shape = [v_shape[0], v_shape[1], capacity, v_shape[3]];
        let mut next_k = mlx_rs::ops::zeros_dtype(&new_k_shape, k.dtype())?;
        let mut next_v = mlx_rs::ops::zeros_dtype(&new_v_shape, v.dtype())?;

        if prev > 0 {
            if let Some(old_k) = self.k.as_ref() {
                let active = if prev >= old_k.shape()[2] {
                    old_k.clone()
                } else {
                    old_k.index((.., .., 0..prev, ..))
                };
                next_k.try_index_mut((.., .., 0..prev, ..), active)?;
            }
            if let Some(old_v) = self.v.as_ref() {
                let active = if prev >= old_v.shape()[2] {
                    old_v.clone()
                } else {
                    old_v.index((.., .., 0..prev, ..))
                };
                next_v.try_index_mut((.., .., 0..prev, ..), active)?;
            }
        }

        self.k = Some(next_k);
        self.v = Some(next_v);
        self.offset = prev;
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FullKvCacheAppendMode {
    TailOwned,
    Concat,
}

fn full_kv_cache_append_mode() -> FullKvCacheAppendMode {
    match std::env::var("FERRITE_FULL_KV_CACHE_APPEND")
        .unwrap_or_default()
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "concat" => FullKvCacheAppendMode::Concat,
        _ => FullKvCacheAppendMode::TailOwned,
    }
}

#[cfg(feature = "native-mlx")]
fn prefill_reserve_enabled() -> bool {
    std::env::var("FERRITE_FULL_KV_CACHE_PREFILL_RESERVE")
        .map(|value| {
            let value = value.trim().to_ascii_lowercase();
            !matches!(value.as_str(), "0" | "false" | "no" | "off")
        })
        .unwrap_or(true)
}

#[derive(Clone, Debug)]
pub struct LinearAttentionCache {
    pub conv_state: Vec<f32>,
    pub recurrent_state: Vec<f32>,
    pub conv_out: Vec<f32>,
    pub recurrent_out: Vec<f32>,
    pub q_normed: Vec<f32>,
    pub k_normed: Vec<f32>,
    pub metal_conv_state: Option<mlx_rs::Array>,
    pub metal_recurrent_state: Option<mlx_rs::Array>,
    pub metal_conv_block_states: Option<mlx_rs::Array>,
    pub metal_recurrent_block_states: Option<mlx_rs::Array>,
}

impl DecodeState {
    pub fn clear_transient_block_states(&mut self) {
        for layer in &mut self.layers {
            if let LayerDecodeState::Linear(cache) = layer {
                cache.metal_conv_block_states = None;
                cache.metal_recurrent_block_states = None;
            }
        }
    }

    pub fn truncate_after_decode_block(
        &mut self,
        base_position: i32,
        keep_tokens: i32,
    ) -> Result<()> {
        if keep_tokens < 0 {
            bail!("cannot keep a negative number of decode tokens: {keep_tokens}");
        }
        let decoded_tokens = self.position - base_position;
        if decoded_tokens < keep_tokens {
            bail!(
                "cannot truncate decode state to {keep_tokens} tokens after base position {base_position}; state only decoded {decoded_tokens}"
            );
        }
        let new_position = base_position + keep_tokens;
        for layer in &mut self.layers {
            match layer {
                LayerDecodeState::Full(cache) => {
                    cache.truncate(new_position);
                }
                LayerDecodeState::Linear(cache) => {
                    if keep_tokens == decoded_tokens {
                        cache.metal_conv_block_states = None;
                        cache.metal_recurrent_block_states = None;
                        continue;
                    }
                    if keep_tokens == 0 {
                        bail!("linear attention decode state cannot truncate back to block start");
                    }
                    let row = keep_tokens - 1;
                    let conv_states = cache
                        .metal_conv_block_states
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing Metal conv block states"))?;
                    let recurrent_states = cache
                        .metal_recurrent_block_states
                        .take()
                        .ok_or_else(|| anyhow::anyhow!("missing Metal recurrent block states"))?;
                    cache.metal_conv_state = Some(conv_states.index((.., row, .., ..)));
                    cache.metal_recurrent_state =
                        Some(recurrent_states.index((.., row, .., .., ..)));
                }
            }
        }
        self.position = new_position;
        Ok(())
    }
}
