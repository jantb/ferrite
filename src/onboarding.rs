use anyhow::Result;
use inquire::{Confirm, Select, Text};

use crate::cli::{DEFAULT_MODEL, DEFAULT_PUBLIC_MODEL_ID, Profile};
use crate::server::ServerConfig;

pub fn server_config(args: &crate::cli::ServeArgs) -> Result<ServerConfig> {
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

    let mode = Select::new(
        "Runtime mode",
        vec![
            ModeChoice::Sustained,
            ModeChoice::SustainedMax,
            ModeChoice::Stable,
            ModeChoice::Burst,
        ],
    )
    .prompt()?;
    let (profile, max) = match mode {
        ModeChoice::Sustained => (Profile::Sustained, false),
        ModeChoice::SustainedMax => (Profile::Sustained, true),
        ModeChoice::Stable => (Profile::Stable, false),
        ModeChoice::Burst => (Profile::PerformanceCold, true),
    };

    let host = Text::new("Bind host").with_default(&args.host).prompt()?;
    let port = Text::new("Bind port")
        .with_default(&args.port.to_string())
        .prompt()?
        .parse::<u16>()?;
    let model_id = Text::new("Served model id")
        .with_default(DEFAULT_PUBLIC_MODEL_ID)
        .prompt()?;
    let max = max
        || Confirm::new("Enable max fan-backed mode?")
            .with_default(args.max)
            .prompt()?;

    Ok(ServerConfig {
        model,
        profile: profile.as_ref_arg().to_string(),
        max,
        host,
        port,
        model_id,
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

#[derive(Clone, Debug)]
enum ModeChoice {
    Sustained,
    SustainedMax,
    Stable,
    Burst,
}

impl std::fmt::Display for ModeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sustained => write!(f, "Sustained: long-context native-MTP"),
            Self::SustainedMax => write!(f, "Sustained Max: Sustained plus fan-backed max mode"),
            Self::Stable => write!(f, "Stable: conservative exact/staged path"),
            Self::Burst => write!(f, "Burst: performance-cold max-fan short-context lane"),
        }
    }
}
