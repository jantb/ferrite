use anyhow::Result;
use inquire::{Select, Text};

use crate::cli::{DEFAULT_MODEL, DEFAULT_PUBLIC_MODEL_ID};
use crate::server::ServerConfig;

pub fn server_config(_args: &crate::cli::ServeArgs) -> Result<ServerConfig> {
    let model_choice = Select::new(
        "Model",
        vec![
            ModelChoice::VerifiedDefault,
            ModelChoice::HuggingFace,
            ModelChoice::LocalPath,
        ],
    )
    .prompt()?;

    let model = match model_choice {
        ModelChoice::VerifiedDefault => DEFAULT_MODEL.to_string(),
        ModelChoice::HuggingFace => Text::new("Hugging Face repo id")
            .with_default(DEFAULT_MODEL)
            .prompt()?,
        ModelChoice::LocalPath => Text::new("Local model path").prompt()?,
    };

    Ok(ServerConfig {
        model,
        host: "127.0.0.1".to_string(),
        port: 11434,
        model_id: DEFAULT_PUBLIC_MODEL_ID.to_string(),
    })
}

pub fn ask_prompt_text() -> Result<String> {
    Ok(Text::new("Prompt").prompt()?)
}

#[derive(Clone, Debug)]
enum ModelChoice {
    VerifiedDefault,
    HuggingFace,
    LocalPath,
}

impl std::fmt::Display for ModelChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VerifiedDefault => write!(f, "Verified default ({DEFAULT_MODEL})"),
            Self::HuggingFace => write!(f, "Custom Hugging Face repo"),
            Self::LocalPath => write!(f, "Local model path"),
        }
    }
}
