use anyhow::Result;
#[cfg(feature = "native-mlx")]
use mlx_rs::ops::indexing::IndexOp;

#[derive(Clone, Copy, Debug)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: u32,
}

#[derive(Clone, Debug)]
pub struct TokenDistribution {
    probabilities: Vec<(u32, f64)>,
}

impl TokenDistribution {
    pub fn deterministic(token: u32) -> Self {
        Self {
            probabilities: vec![(token, 1.0)],
        }
    }

    pub fn probability(&self, token: u32) -> f64 {
        self.probabilities
            .iter()
            .find_map(|(id, probability)| (*id == token).then_some(*probability))
            .unwrap_or(0.0)
    }

    pub fn sample(&self) -> Result<u32> {
        if self.probabilities.is_empty() {
            anyhow::bail!("cannot sample from empty distribution");
        }
        let mut draw = rand::random::<f64>();
        for (id, probability) in &self.probabilities {
            draw -= *probability;
            if draw <= 0.0 {
                return Ok(*id);
            }
        }
        Ok(self.probabilities[self.probabilities.len() - 1].0)
    }

    pub fn residual_from(target: &Self, draft: &Self) -> Self {
        let mut probabilities = target
            .probabilities
            .iter()
            .filter_map(|(id, target_probability)| {
                let probability = target_probability - draft.probability(*id);
                (probability > 0.0).then_some((*id, probability))
            })
            .collect::<Vec<_>>();
        normalize_probabilities(&mut probabilities);
        if probabilities.is_empty() {
            return target.clone();
        }
        Self { probabilities }
    }
}

impl SamplingConfig {
    pub(crate) fn is_greedy(self) -> bool {
        self.temperature <= 0.0 || self.top_k == 1
    }
}

#[cfg(feature = "native-mlx")]
pub fn next_from_logits(logits: &mlx_rs::Array, config: SamplingConfig) -> Result<u32> {
    distribution_from_logits(logits, config)?.sample()
}

#[cfg(feature = "native-mlx")]
pub fn distribution_from_logits(
    logits: &mlx_rs::Array,
    config: SamplingConfig,
) -> Result<TokenDistribution> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 {
        anyhow::bail!("unexpected logits shape: {shape:?}");
    }
    let vocab = shape[2] as usize;
    if config.is_greedy() {
        return Ok(TokenDistribution::deterministic(greedy_token_from_logits(
            logits,
        )?));
    }
    if cpu_topk_sampling_enabled() {
        let values = crate::qwen36::array_to_f32_vec(logits)?;
        return distribution_from_row(&values[..vocab], config);
    }
    if config.top_k > 0 && (config.top_k as usize) < vocab {
        return distribution_from_logits_topk(logits, config, vocab, config.top_k as usize);
    }
    let values = crate::qwen36::array_to_f32_vec(logits)?;
    distribution_from_row(&values[..vocab], config)
}

#[cfg(feature = "native-mlx")]
pub fn distributions_from_logits(
    logits: &mlx_rs::Array,
    config: SamplingConfig,
) -> Result<Vec<TokenDistribution>> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] < 1 {
        anyhow::bail!("unexpected logits shape: {shape:?}");
    }
    let tokens = shape[1] as usize;
    if config.is_greedy() {
        return greedy_tokens_from_logits(logits).map(|tokens| {
            tokens
                .into_iter()
                .map(TokenDistribution::deterministic)
                .collect()
        });
    }
    let vocab = shape[2] as usize;
    if cpu_topk_sampling_enabled() {
        let values = crate::qwen36::array_to_f32_vec(logits)?;
        return values
            .chunks_exact(vocab)
            .take(tokens)
            .map(|row| distribution_from_row(row, config))
            .collect();
    }
    if config.top_k > 0 && (config.top_k as usize) < vocab {
        return distributions_from_logits_topk(logits, config, vocab, config.top_k as usize);
    }
    let values = crate::qwen36::array_to_f32_vec(logits)?;
    values
        .chunks_exact(vocab)
        .take(tokens)
        .map(|row| distribution_from_row(row, config))
        .collect()
}

#[cfg(feature = "native-mlx")]
fn cpu_topk_sampling_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("MTPLX_CPU_TOPK_SAMPLING")
            .ok()
            .map(|value| {
                let value = value.trim().to_ascii_lowercase();
                matches!(value.as_str(), "1" | "true" | "yes" | "on")
            })
            .unwrap_or(false)
    })
}

#[cfg(feature = "native-mlx")]
fn distribution_from_logits_topk(
    logits: &mlx_rs::Array,
    config: SamplingConfig,
    vocab: usize,
    top_k: usize,
) -> Result<TokenDistribution> {
    let start = vocab as i32 - top_k as i32;
    let partition = mlx_rs::ops::argpartition_axis(logits, start, -1)?;
    let indices = partition.index((.., .., start..));
    let values = logits.take_along_axis(&indices, Some(-1))?;
    let indices = indices.as_type::<u32>()?;
    let values = values.as_type::<f32>()?;
    indices.eval()?;
    values.eval()?;
    let mut candidates = indices
        .as_slice::<u32>()
        .iter()
        .copied()
        .zip(values.as_slice::<f32>().iter().copied())
        .collect::<Vec<_>>();
    distribution_from_candidates(&mut candidates, config)
}

#[cfg(feature = "native-mlx")]
fn distributions_from_logits_topk(
    logits: &mlx_rs::Array,
    config: SamplingConfig,
    vocab: usize,
    top_k: usize,
) -> Result<Vec<TokenDistribution>> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] < 1 {
        anyhow::bail!("unexpected logits shape: {shape:?}");
    }
    let tokens = shape[1] as usize;
    let start = vocab as i32 - top_k as i32;
    let partition = mlx_rs::ops::argpartition_axis(logits, start, -1)?;
    let indices = partition.index((.., .., start..));
    let values = logits.take_along_axis(&indices, Some(-1))?;
    let expected = tokens * top_k;
    let indices = indices.as_type::<u32>()?.reshape(&[expected as i32])?;
    let values = values.as_type::<f32>()?.reshape(&[expected as i32])?;
    indices.eval()?;
    values.eval()?;
    let indices = indices.as_slice::<u32>();
    let values = values.as_slice::<f32>();
    if indices.len() < expected || values.len() < expected {
        anyhow::bail!(
            "unexpected top-k result length: indices={} values={} expected={expected}",
            indices.len(),
            values.len()
        );
    }

    (0..tokens)
        .map(|row| {
            let offset = row * top_k;
            let mut candidates = indices[offset..offset + top_k]
                .iter()
                .copied()
                .zip(values[offset..offset + top_k].iter().copied())
                .collect::<Vec<_>>();
            distribution_from_candidates(&mut candidates, config)
        })
        .collect()
}

#[cfg(feature = "native-mlx")]
pub fn greedy_token_from_logits(logits: &mlx_rs::Array) -> Result<u32> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 {
        anyhow::bail!("unexpected logits shape: {shape:?}");
    }
    let id = mlx_rs::ops::indexing::argmax_axis(logits, -1, false)?.as_type::<u32>()?;
    id.eval()?;
    Ok(id.as_slice::<u32>()[0])
}

#[cfg(feature = "native-mlx")]
pub fn greedy_tokens_from_logits(logits: &mlx_rs::Array) -> Result<Vec<u32>> {
    let shape = logits.shape().to_vec();
    if shape.len() != 3 || shape[0] != 1 {
        anyhow::bail!("unexpected logits shape: {shape:?}");
    }
    let ids = mlx_rs::ops::indexing::argmax_axis(logits, -1, false)?.as_type::<u32>()?;
    ids.eval()?;
    Ok(ids.as_slice::<u32>().to_vec())
}

#[allow(dead_code)]
pub fn select_from_row(row: &[f32], config: SamplingConfig) -> Result<u32> {
    distribution_from_row(row, config)?.sample()
}

pub fn distribution_from_row(row: &[f32], config: SamplingConfig) -> Result<TokenDistribution> {
    if row.is_empty() {
        anyhow::bail!("cannot sample from empty logits row");
    }
    if config.is_greedy() {
        return Ok(TokenDistribution {
            probabilities: vec![(argmax(row) as u32, 1.0)],
        });
    }

    let mut candidates = top_k_candidates(row, config.top_k);
    let mut candidates = candidates
        .drain(..)
        .map(|(id, value)| (id as u32, value))
        .collect::<Vec<_>>();
    distribution_from_candidates(&mut candidates, config)
}

fn distribution_from_candidates(
    candidates: &mut Vec<(u32, f32)>,
    config: SamplingConfig,
) -> Result<TokenDistribution> {
    if candidates.is_empty() {
        anyhow::bail!("cannot sample from empty candidate set");
    }
    let temperature = config.temperature.max(1.0e-6);
    let max_logit = candidates
        .iter()
        .map(|(_, value)| *value)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut total = 0.0_f64;
    for (_, value) in candidates.iter_mut() {
        let weight = ((*value - max_logit) / temperature).exp() as f64;
        *value = weight as f32;
        total += weight;
    }
    if !total.is_finite() || total <= 0.0 {
        let id = candidates
            .iter()
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .map(|(id, _)| *id)
            .unwrap_or(0);
        return Ok(TokenDistribution {
            probabilities: vec![(id, 1.0)],
        });
    }

    candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
    apply_top_p(candidates, config.top_p, total);
    let mut probabilities = candidates
        .iter()
        .map(|(id, weight)| (*id, f64::from(*weight)))
        .collect::<Vec<_>>();
    normalize_probabilities(&mut probabilities);
    Ok(TokenDistribution { probabilities })
}

fn argmax(row: &[f32]) -> usize {
    let mut best_id = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (id, value) in row.iter().copied().enumerate() {
        if value > best_value {
            best_value = value;
            best_id = id;
        }
    }
    best_id
}

fn top_k_candidates(row: &[f32], top_k: u32) -> Vec<(usize, f32)> {
    let limit = if top_k == 0 {
        row.len()
    } else {
        (top_k as usize).min(row.len())
    };
    if limit == row.len() {
        return row.iter().copied().enumerate().collect();
    }
    let mut candidates = Vec::with_capacity(limit);
    for (id, value) in row.iter().copied().enumerate() {
        if candidates.len() < limit {
            candidates.push((id, value));
            if candidates.len() == limit {
                candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
            }
            continue;
        }
        if value <= candidates[limit - 1].1 {
            continue;
        }
        let mut insert_at = limit - 1;
        while insert_at > 0 && value > candidates[insert_at - 1].1 {
            candidates[insert_at] = candidates[insert_at - 1];
            insert_at -= 1;
        }
        candidates[insert_at] = (id, value);
    }
    candidates
}

fn apply_top_p<T: Copy>(candidates: &mut Vec<(T, f32)>, top_p: f32, total: f64) -> f64 {
    let top_p = top_p.clamp(0.0, 1.0);
    if top_p >= 1.0 {
        return total;
    }
    let threshold = total * f64::from(top_p.max(1.0e-6));
    let mut kept = 0usize;
    let mut cumulative = 0.0_f64;
    for (_, weight) in candidates.iter() {
        kept += 1;
        cumulative += f64::from(*weight);
        if cumulative >= threshold {
            break;
        }
    }
    candidates.truncate(kept.max(1));
    candidates
        .iter()
        .map(|(_, weight)| f64::from(*weight))
        .sum()
}

fn normalize_probabilities(probabilities: &mut Vec<(u32, f64)>) {
    let total = probabilities
        .iter()
        .map(|(_, probability)| *probability)
        .sum::<f64>();
    if !total.is_finite() || total <= 0.0 {
        probabilities.clear();
        return;
    }
    for (_, probability) in probabilities {
        *probability /= total;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temperature_zero_selects_argmax() {
        let id = select_from_row(
            &[1.0, 3.0, 2.0],
            SamplingConfig {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 20,
            },
        )
        .unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn top_k_one_selects_argmax() {
        let id = select_from_row(
            &[1.0, 3.0, 2.0],
            SamplingConfig {
                temperature: 0.7,
                top_p: 1.0,
                top_k: 1,
            },
        )
        .unwrap();
        assert_eq!(id, 1);
    }

    #[test]
    fn top_k_candidates_keeps_only_best_values() {
        let candidates = top_k_candidates(&[1.0, 9.0, 2.0, 8.0, 3.0], 2);
        assert_eq!(candidates, vec![(1, 9.0), (3, 8.0)]);
    }

    #[cfg(feature = "native-mlx")]
    #[test]
    fn mlx_top_k_distributions_match_cpu_rows() {
        let _guard = crate::mlx_test_lock();
        let config = SamplingConfig {
            temperature: 0.6,
            top_p: 1.0,
            top_k: 3,
        };
        let rows = [
            [1.0_f32, 7.0, -2.0, 4.0, 9.0, 6.0],
            [8.0_f32, -1.0, 3.0, 5.0, 2.0, 10.0],
            [0.0_f32, 12.0, 11.0, 1.0, -3.0, 4.0],
        ];
        let flat = rows.iter().flatten().copied().collect::<Vec<_>>();
        let logits = mlx_rs::Array::from_slice(&flat, &[1, rows.len() as i32, 6]);
        let single_row_logits = mlx_rs::Array::from_slice(&rows[0], &[1, 1, 6]);

        let single_row_mlx = distribution_from_logits(&single_row_logits, config).unwrap();
        let single_row_cpu = distribution_from_row(&rows[0], config).unwrap();
        let mlx = distributions_from_logits(&logits, config).unwrap();

        assert_distribution_eq(&single_row_mlx, &single_row_cpu, "single row");
        assert_eq!(mlx.len(), rows.len());
        for (index, row) in rows.iter().enumerate() {
            let cpu = distribution_from_row(row, config).unwrap();
            assert_distribution_eq(&mlx[index], &cpu, &format!("row {index}"));
        }
    }

    fn assert_distribution_eq(left: &TokenDistribution, right: &TokenDistribution, label: &str) {
        assert_eq!(
            left.probabilities.len(),
            right.probabilities.len(),
            "{label}"
        );
        for ((left_id, left_probability), (right_id, right_probability)) in
            left.probabilities.iter().zip(right.probabilities.iter())
        {
            assert_eq!(left_id, right_id, "{label}");
            assert!(
                (left_probability - right_probability).abs() < 1.0e-6,
                "{label} token {left_id}: left={left_probability} right={right_probability}"
            );
        }
    }
}
