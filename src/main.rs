mod api;
mod artifacts;
mod bench;
mod cli;
mod inference;
#[cfg(feature = "native-mlx")]
mod metal_kernels;
#[cfg(feature = "native-mlx")]
mod mlx_backend;
mod model;
mod onboarding;
mod qwen36;
mod sampling;
mod server;
mod state;

use anyhow::Result;
use clap::Parser;

#[cfg(all(test, feature = "native-mlx"))]
static MLX_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(test, feature = "native-mlx"))]
pub(crate) fn mlx_test_lock() -> std::sync::MutexGuard<'static, ()> {
    MLX_TEST_LOCK.lock().unwrap()
}

fn main() -> Result<()> {
    let cli = cli::Cli::parse();
    cli::run(cli)
}
