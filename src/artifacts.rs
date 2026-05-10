use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use safetensors::SafeTensors;
use serde::{Deserialize, Serialize};

use crate::model::ModelConfig;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorInfo {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub source_file: PathBuf,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelTensors {
    pub model_tensors: BTreeMap<String, TensorInfo>,
    pub mtp_tensors: BTreeMap<String, TensorInfo>,
}

#[derive(Debug, Deserialize)]
struct SafetensorsIndex {
    weight_map: BTreeMap<String, String>,
}

pub fn inspect_tensors(model_dir: &Path, config: &ModelConfig) -> Result<ModelTensors> {
    let mut model_tensors = BTreeMap::new();
    let mut files = BTreeSet::new();
    let index_path = model_dir.join("model.safetensors.index.json");
    if index_path.is_file() {
        let text = std::fs::read_to_string(&index_path)
            .with_context(|| format!("read {}", index_path.display()))?;
        let index: SafetensorsIndex = serde_json::from_str(&text)
            .with_context(|| format!("parse {}", index_path.display()))?;
        for file in index.weight_map.values() {
            files.insert(file.clone());
        }
    } else {
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) == Some("safetensors")
                && path.file_name().and_then(|name| name.to_str()) != Some("mtp.safetensors")
            {
                if let Some(name) = path.file_name().and_then(|name| name.to_str()) {
                    files.insert(name.to_string());
                }
            }
        }
    }

    for file in files {
        let path = model_dir.join(&file);
        if path.is_file() {
            inspect_file(&path, &mut model_tensors)?;
        }
    }

    let mut mtp_tensors = BTreeMap::new();
    let mtp_path = expected_mtp_file(model_dir, config);
    if mtp_path.is_file() {
        inspect_file(&mtp_path, &mut mtp_tensors)?;
    }

    Ok(ModelTensors {
        model_tensors,
        mtp_tensors,
    })
}

pub fn expected_mtp_file(model_dir: &Path, config: &ModelConfig) -> PathBuf {
    if let Some(value) = config
        .mlx_lm_extra_tensors
        .get("mtp_file")
        .and_then(|value| value.as_str())
    {
        return model_dir.join(value);
    }
    for rel in [
        "mtp.safetensors",
        "mtp/weights.safetensors",
        "model-mtp.safetensors",
    ] {
        let candidate = model_dir.join(rel);
        if candidate.exists() {
            return candidate;
        }
    }
    model_dir.join("mtp.safetensors")
}

fn inspect_file(path: &Path, out: &mut BTreeMap<String, TensorInfo>) -> Result<()> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    let tensors = SafeTensors::deserialize(&bytes)
        .map_err(|err| anyhow::anyhow!("parse safetensors {}: {err}", path.display()))?;
    for (name, view) in tensors.tensors() {
        out.insert(
            name.to_string(),
            TensorInfo {
                name: name.to_string(),
                dtype: format!("{:?}", view.dtype()),
                shape: view.shape().to_vec(),
                source_file: path.to_path_buf(),
            },
        );
    }
    Ok(())
}
