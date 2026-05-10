use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

const DEFAULT_MODEL_CACHE: &str = ".mtplx/models";

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelConfig {
    #[serde(default)]
    pub architectures: Vec<String>,
    pub model_type: Option<String>,
    pub text_config: Option<Box<ModelConfig>>,
    pub hidden_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub vocab_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub mtp_num_hidden_layers: Option<u32>,
    pub mtp_pattern: Option<String>,
    #[serde(default)]
    pub mlx_lm_extra_tensors: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub mtplx_mtp_quantization: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub quantization: BTreeMap<String, serde_json::Value>,
    pub rms_norm_eps: Option<f32>,
    pub head_dim: Option<u32>,
    pub num_attention_heads: Option<u32>,
    pub num_key_value_heads: Option<u32>,
    pub partial_rotary_factor: Option<f32>,
    pub linear_key_head_dim: Option<u32>,
    pub linear_value_head_dim: Option<u32>,
    pub linear_num_key_heads: Option<u32>,
    pub linear_num_value_heads: Option<u32>,
    #[serde(default)]
    pub rope_parameters: BTreeMap<String, serde_json::Value>,
    pub eos_token_id: Option<serde_json::Value>,
}

impl ModelConfig {
    pub fn text(&self) -> &ModelConfig {
        self.text_config.as_deref().unwrap_or(self)
    }

    pub fn architecture(&self) -> Option<&str> {
        self.text()
            .architectures
            .first()
            .map(String::as_str)
            .or_else(|| self.text().model_type.as_deref())
    }
}

#[derive(Debug)]
pub struct LoadedModel {
    pub path: PathBuf,
    pub config: ModelConfig,
    pub runtime_contract: Option<RuntimeContract>,
    pub tokenizer: Tokenizer,
    pub tensors: crate::artifacts::ModelTensors,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeContract {
    pub recommended_draft_lm_head: Option<DraftLmHeadSpec>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DraftLmHeadSpec {
    pub bits: i32,
    #[serde(default = "default_draft_lm_head_group_size")]
    pub group_size: i32,
    #[serde(default = "default_draft_lm_head_mode")]
    pub mode: String,
}

fn default_draft_lm_head_group_size() -> i32 {
    64
}

fn default_draft_lm_head_mode() -> String {
    "affine".to_string()
}

#[derive(Clone, Debug, Serialize)]
pub struct ModelSummary {
    pub path: PathBuf,
    pub architecture: Option<String>,
    pub model_type: Option<String>,
    pub hidden_size: Option<u32>,
    pub num_hidden_layers: Option<u32>,
    pub vocab_size: Option<u32>,
    pub intermediate_size: Option<u32>,
    pub mtp_num_hidden_layers: Option<u32>,
    pub model_tensor_count: usize,
    pub mtp_tensor_count: usize,
    pub mtp_present: bool,
    pub tokenizer_present: bool,
}

impl LoadedModel {
    pub fn load(model_ref: &str) -> Result<Self> {
        let path = resolve_model_path(model_ref)?;
        let config = load_config(&path)?;
        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.is_file() {
            bail!("missing tokenizer.json in {}", path.display());
        }
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|err| anyhow::anyhow!("load tokenizer {}: {err}", tokenizer_path.display()))?;
        let tensors = crate::artifacts::inspect_tensors(&path, &config)?;
        let runtime_contract = load_runtime_contract(&path)?;
        Ok(Self {
            path,
            config,
            runtime_contract,
            tokenizer,
            tensors,
        })
    }

    pub fn encode_prompt(&self, request: &crate::inference::InferenceRequest) -> Result<Vec<u32>> {
        let text = self.render_chat_prompt(request);
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("tokenize prompt: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn encode_chat_prefix(
        &self,
        system: Option<&str>,
        messages: &[crate::inference::PromptMessage],
    ) -> Result<Vec<u32>> {
        let text = self.render_chat_prefix(system, messages);
        let encoding = self
            .tokenizer
            .encode(text, false)
            .map_err(|err| anyhow::anyhow!("tokenize chat prefix: {err}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    pub fn render_chat_prompt(&self, request: &crate::inference::InferenceRequest) -> String {
        let owned_messages;
        let messages = if request.messages.is_empty() {
            owned_messages = vec![crate::inference::PromptMessage {
                role: "user".to_string(),
                content: request.prompt.clone(),
            }];
            &owned_messages
        } else {
            &request.messages
        };
        let mut text = self.render_chat_prefix(request.system.as_deref(), messages);
        text.push_str("<|im_start|>assistant\n");
        text.push_str("<think>\n\n</think>\n\n");
        text
    }

    pub fn render_chat_prefix(
        &self,
        system: Option<&str>,
        messages: &[crate::inference::PromptMessage],
    ) -> String {
        let mut text = String::new();
        if let Some(system) = system {
            append_chat_message(&mut text, "system", system);
        }
        for message in messages {
            append_chat_message(&mut text, &message.role, &message.content);
        }
        text
    }

    pub fn eos_token_ids(&self) -> Vec<u32> {
        let mut ids = Vec::new();
        collect_token_ids(self.config.text().eos_token_id.as_ref(), &mut ids);
        let path = self.path.join("generation_config.json");
        if let Ok(text) = std::fs::read_to_string(path) {
            if let Ok(value) = serde_json::from_str::<serde_json::Value>(&text) {
                collect_token_ids(value.get("eos_token_id"), &mut ids);
            }
        }
        ids.sort_unstable();
        ids.dedup();
        ids
    }

    pub fn summary(&self) -> ModelSummary {
        let text = self.config.text();
        ModelSummary {
            path: self.path.clone(),
            architecture: self.config.architecture().map(str::to_string),
            model_type: text.model_type.clone(),
            hidden_size: text.hidden_size,
            num_hidden_layers: text.num_hidden_layers,
            vocab_size: text.vocab_size,
            intermediate_size: text.intermediate_size,
            mtp_num_hidden_layers: text.mtp_num_hidden_layers,
            model_tensor_count: self.tensors.model_tensors.len(),
            mtp_tensor_count: self.tensors.mtp_tensors.len(),
            mtp_present: !self.tensors.mtp_tensors.is_empty(),
            tokenizer_present: true,
        }
    }
}

fn append_chat_message(text: &mut String, role: &str, content: &str) {
    text.push_str("<|im_start|>");
    text.push_str(role.trim());
    text.push('\n');
    text.push_str(content.trim());
    text.push_str("<|im_end|>\n");
}

fn collect_token_ids(value: Option<&serde_json::Value>, out: &mut Vec<u32>) {
    match value {
        Some(serde_json::Value::Number(number)) => {
            if let Some(id) = number.as_u64().and_then(|id| u32::try_from(id).ok()) {
                out.push(id);
            }
        }
        Some(serde_json::Value::Array(items)) => {
            for item in items {
                collect_token_ids(Some(item), out);
            }
        }
        _ => {}
    }
}

pub fn load_config(model_dir: &Path) -> Result<ModelConfig> {
    let path = model_dir.join("config.json");
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

pub fn load_runtime_contract(model_dir: &Path) -> Result<Option<RuntimeContract>> {
    let path = model_dir.join("mtplx_runtime.json");
    if !path.is_file() {
        return Ok(None);
    }
    let text =
        std::fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let contract =
        serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(contract))
}

pub fn resolve_model_path(model_ref: &str) -> Result<PathBuf> {
    let direct = expand_tilde(model_ref);
    if direct.exists() {
        return Ok(direct);
    }
    if let Some(repo_id) = repo_id_from_ref(model_ref) {
        let cached = model_cache_dir().join(safe_model_name(repo_id));
        if cached.join("config.json").is_file() {
            return Ok(cached);
        }
    }
    bail!("model is not available locally: {model_ref}");
}

pub fn model_cache_dir() -> PathBuf {
    if let Some(value) =
        std::env::var_os("FERRITE_MODEL_DIR").or_else(|| std::env::var_os("MTPLX_MODEL_DIR"))
    {
        return expand_tilde(&value.to_string_lossy());
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(DEFAULT_MODEL_CACHE)
}

pub fn repo_id_from_ref(value: &str) -> Option<&str> {
    let trimmed = value.trim();
    if trimmed.starts_with('/') || trimmed.starts_with('~') || trimmed.starts_with('.') {
        return None;
    }
    let mut parts = trimmed.split('/');
    match (parts.next(), parts.next(), parts.next()) {
        (Some(namespace), Some(name), None)
            if valid_repo_part(namespace) && valid_repo_part(name) =>
        {
            Some(trimmed)
        }
        _ => None,
    }
}

pub fn safe_model_name(repo_id: &str) -> String {
    repo_id.trim_matches('/').replace('/', "--")
}

pub fn expand_tilde(value: &str) -> PathBuf {
    if let Some(rest) = value.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest);
        }
    }
    PathBuf::from(value)
}

fn valid_repo_part(value: &str) -> bool {
    !value.is_empty()
        && value
            .chars()
            .all(|ch| ch.is_ascii_alphanumeric() || matches!(ch, '_' | '-' | '.'))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recognizes_hf_repo_ids() {
        assert_eq!(
            repo_id_from_ref("Youssofal/Qwen3.6"),
            Some("Youssofal/Qwen3.6")
        );
        assert_eq!(repo_id_from_ref("/tmp/model"), None);
        assert_eq!(repo_id_from_ref("one/two/three"), None);
    }

    #[test]
    fn safe_model_name_replaces_slash() {
        assert_eq!(safe_model_name("a/b"), "a--b");
    }
}
