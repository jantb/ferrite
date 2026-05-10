use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::inference::{InferenceRequest, PromptMessage};

#[derive(Clone, Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop: Option<Value>,
    pub stream: Option<bool>,
    pub generation_mode: Option<String>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CompletionRequest {
    pub model: Option<String>,
    pub prompt: Value,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub stop: Option<Value>,
    pub stream: Option<bool>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: Value,
}

impl ChatCompletionRequest {
    pub fn into_inference(self, default_model: &str, public_model_id: &str) -> InferenceRequest {
        let mut system = None;
        let mut prompt_parts = Vec::new();
        let mut messages = Vec::new();
        for message in self.messages {
            let content = content_to_text(&message.content);
            if message.role == "system" {
                system = Some(content);
            } else {
                if message.role == "user" {
                    prompt_parts.push(content.clone());
                } else {
                    prompt_parts.push(format!("{}: {}", message.role, content));
                }
                messages.push(PromptMessage {
                    role: message.role,
                    content,
                });
            }
        }
        let model = match self.model {
            Some(model) if !is_public_model_alias(&model, public_model_id) => model,
            _ => default_model.to_string(),
        };
        InferenceRequest {
            model,
            prompt: prompt_parts.join("\n"),
            system,
            messages,
            stop: stop_to_strings(self.stop.as_ref()),
            max_tokens: self.max_tokens.or(self.max_completion_tokens),
            temperature: self.temperature.unwrap_or(0.6),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k.unwrap_or(20),
            depth: 2,
            mtp: self.generation_mode.as_deref() != Some("ar"),
            profile_timings: false,
        }
    }
}

impl CompletionRequest {
    pub fn into_inference(self, default_model: &str, public_model_id: &str) -> InferenceRequest {
        let model = match self.model {
            Some(model) if !is_public_model_alias(&model, public_model_id) => model,
            _ => default_model.to_string(),
        };
        InferenceRequest {
            model,
            prompt: prompt_to_text(&self.prompt),
            system: None,
            messages: Vec::new(),
            stop: stop_to_strings(self.stop.as_ref()),
            max_tokens: self.max_tokens,
            temperature: self.temperature.unwrap_or(0.6),
            top_p: self.top_p.unwrap_or(0.95),
            top_k: self.top_k.unwrap_or(20),
            depth: 2,
            mtp: true,
            profile_timings: false,
        }
    }
}

fn is_public_model_alias(model: &str, public_model_id: &str) -> bool {
    model == public_model_id || model == "mtplx-qwen36-27b-optimized-speed"
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorResponse {
    pub error: ErrorBody,
}

#[derive(Clone, Debug, Serialize)]
pub struct ErrorBody {
    pub message: String,
    #[serde(rename = "type")]
    pub kind: String,
}

pub fn error(message: impl Into<String>, kind: impl Into<String>) -> ErrorResponse {
    ErrorResponse {
        error: ErrorBody {
            message: message.into(),
            kind: kind.into(),
        },
    }
}

pub fn content_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .filter_map(|item| {
                item.get("text")
                    .and_then(Value::as_str)
                    .or_else(|| item.get("content").and_then(Value::as_str))
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

pub fn prompt_to_text(value: &Value) -> String {
    match value {
        Value::String(text) => text.clone(),
        Value::Array(items) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| item.to_string())
            })
            .collect::<Vec<_>>()
            .join("\n"),
        other => other.to_string(),
    }
}

pub fn stop_to_strings(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::String(text)) => vec![text.clone()],
        Some(Value::Array(items)) => items
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect(),
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_model_id_maps_to_configured_model() {
        let request = ChatCompletionRequest {
            model: Some("public-id".to_string()),
            messages: vec![ChatMessage {
                role: "user".to_string(),
                content: Value::String("hi".to_string()),
            }],
            max_tokens: Some(4),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: None,
            generation_mode: None,
        };

        let inference = request.into_inference("local/model", "public-id");
        assert_eq!(inference.model, "local/model");
        assert_eq!(inference.prompt, "hi");
        assert_eq!(inference.messages.len(), 1);
        assert_eq!(inference.messages[0].role, "user");
        assert_eq!(inference.messages[0].content, "hi");
        assert_eq!(inference.temperature, 0.6);
        assert_eq!(inference.top_p, 0.95);
        assert_eq!(inference.top_k, 20);
    }

    #[test]
    fn completion_request_maps_prompt_and_sampling_defaults() {
        let request = CompletionRequest {
            model: Some("public-id".to_string()),
            prompt: Value::String("hi".to_string()),
            max_tokens: Some(4),
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: None,
        };

        let inference = request.into_inference("local/model", "public-id");
        assert_eq!(inference.model, "local/model");
        assert_eq!(inference.prompt, "hi");
        assert_eq!(inference.temperature, 0.6);
        assert_eq!(inference.top_p, 0.95);
        assert_eq!(inference.top_k, 20);
    }

    #[test]
    fn chat_request_preserves_assistant_turns() {
        let request = ChatCompletionRequest {
            model: None,
            messages: vec![
                ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("hi".to_string()),
                },
                ChatMessage {
                    role: "assistant".to_string(),
                    content: Value::String("Hello!".to_string()),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: Value::String("again".to_string()),
                },
            ],
            max_tokens: Some(4),
            max_completion_tokens: None,
            temperature: None,
            top_p: None,
            top_k: None,
            stop: None,
            stream: None,
            generation_mode: None,
        };

        let inference = request.into_inference("local/model", "public-id");
        assert_eq!(inference.messages.len(), 3);
        assert_eq!(inference.messages[1].role, "assistant");
        assert_eq!(inference.messages[1].content, "Hello!");
        assert_eq!(inference.prompt, "hi\nassistant: Hello!\nagain");
    }
}
