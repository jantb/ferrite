use std::fs;
use std::path::PathBuf;

use anyhow::{Context, Result};

use crate::server::ServerConfig;

pub fn load_server_config() -> Result<Option<ServerConfig>> {
    let path = state_path();
    if !path.exists() {
        return Ok(None);
    }
    let text = fs::read_to_string(&path).with_context(|| format!("read {}", path.display()))?;
    let state = serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))?;
    Ok(Some(state))
}

pub fn save_server_config(state: &ServerConfig) -> Result<()> {
    let path = state_path();
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let text = serde_json::to_string_pretty(state)?;
    fs::write(&path, format!("{text}\n")).with_context(|| format!("write {}", path.display()))
}

pub fn state_path() -> PathBuf {
    if let Some(path) =
        std::env::var_os("FERRITE_STATE").or_else(|| std::env::var_os("MTPLX_RS_QUICKSTART_STATE"))
    {
        return PathBuf::from(path);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ferrite")
        .join("server.json")
}
