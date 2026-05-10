use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};

use crate::inference::{InferenceBackend, InferenceRequest};
use crate::onboarding;
use crate::server::ServerConfig;

pub const DEFAULT_MODEL: &str = "Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed";
pub const DEFAULT_PUBLIC_MODEL_ID: &str = "ferrite-qwen36-27b-optimized-speed";

#[derive(Parser, Debug)]
#[command(name = "ferrite")]
#[command(about = "Native MTP inference for Apple Silicon agents")]
#[command(version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Option<Command>,
}

#[derive(Subcommand, Debug)]
pub enum Command {
    /// Start the OpenAI-compatible inference server.
    Serve(ServeArgs),
    /// Run one local inference request through the native Rust backend.
    Infer(InferArgs),
    /// Inspect a local model without running tensor inference.
    Inspect(InspectArgs),
    /// Benchmark the currently ported native MLX Rust path.
    BenchNative(BenchNativeArgs),
    /// Benchmark end-to-end generated output tokens.
    BenchDecode(BenchDecodeArgs),
    /// Benchmark the same inference path used by the server.
    BenchInfer(BenchInferArgs),
    /// Benchmark greedy MTP draft acceptance against verified target decode.
    BenchMtp(BenchMtpArgs),
    /// Probe usable long-context prefill/decode windows.
    BenchContext(BenchContextArgs),
    /// Inspect Rust server state.
    Status(StatusArgs),
}

#[derive(Args, Debug)]
pub struct ServeArgs {
    /// Model path or Hugging Face repo id.
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "127.0.0.1")]
    pub host: String,
    #[arg(long, default_value_t = 11434)]
    pub port: u16,
    #[arg(long, default_value = DEFAULT_PUBLIC_MODEL_ID)]
    pub model_id: String,
    /// Ask for missing server setup with inquire.
    #[arg(long)]
    pub interactive: bool,
    #[arg(long)]
    pub dry_run: bool,
}

#[derive(Args, Debug)]
pub struct InferArgs {
    pub prompt: Option<String>,
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long)]
    pub system: Option<String>,
    #[arg(long)]
    pub max_tokens: Option<u32>,
    #[arg(long, default_value_t = 0.6)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 20)]
    pub top_k: u32,
    #[arg(long = "stop")]
    pub stop: Vec<String>,
    #[arg(long, default_value_t = 2)]
    pub depth: u32,
    #[arg(long)]
    pub no_mtp: bool,
    /// Force MLX eval at timing boundaries. Slower, but useful for profiling.
    #[arg(long)]
    pub profile_timings: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct StatusArgs {
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct InspectArgs {
    pub model: Option<String>,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchNativeArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "hi")]
    pub prompt: String,
    #[arg(long, default_value_t = 20)]
    pub iterations: u32,
    #[arg(long, default_value_t = 3)]
    pub warmup: u32,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchDecodeArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "hi")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub tokens: u32,
    #[arg(long, default_value_t = 0.6)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 20)]
    pub top_k: u32,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchInferArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "hi")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    #[arg(long, default_value_t = 1)]
    pub prompt_repeat: usize,
    #[arg(long, default_value_t = 1)]
    pub tokens: u32,
    #[arg(long, default_value_t = 0.0)]
    pub temperature: f32,
    #[arg(long, default_value_t = 1.0)]
    pub top_p: f32,
    #[arg(long, default_value_t = 1)]
    pub top_k: u32,
    #[arg(long, default_value_t = 1)]
    pub iterations: u32,
    #[arg(long, default_value_t = 0)]
    pub warmup: u32,
    #[arg(long, default_value_t = 2)]
    pub depth: u32,
    #[arg(long)]
    pub mtp: bool,
    #[arg(long)]
    pub profile_timings: bool,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchMtpArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(long, default_value = "hi")]
    pub prompt: String,
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    #[arg(long, default_value_t = 16)]
    pub tokens: u32,
    #[arg(long, default_value_t = 2)]
    pub depth: u32,
    #[arg(long)]
    pub json: bool,
}

#[derive(Args, Debug)]
pub struct BenchContextArgs {
    #[arg(long)]
    pub model: Option<String>,
    #[arg(
        long,
        default_value = "You are a coding agent. Inspect the repository, make focused edits, and verify your work."
    )]
    pub prompt: String,
    #[arg(long)]
    pub prompt_file: Option<PathBuf>,
    /// Comma-separated prompt token counts to test.
    #[arg(long, default_value = "8192,16384,32768")]
    pub contexts: String,
    #[arg(long, default_value_t = 4)]
    pub generate_tokens: u32,
    #[arg(long, default_value_t = 0.6)]
    pub temperature: f32,
    #[arg(long, default_value_t = 0.95)]
    pub top_p: f32,
    #[arg(long, default_value_t = 20)]
    pub top_k: u32,
    #[arg(long)]
    pub json: bool,
}

pub fn run(cli: Cli) -> Result<()> {
    match cli.command.unwrap_or(Command::Serve(ServeArgs::default())) {
        Command::Serve(args) => serve(args),
        Command::Infer(args) => infer(args),
        Command::Inspect(args) => inspect(args),
        Command::BenchNative(args) => bench_native(args),
        Command::BenchDecode(args) => bench_decode(args),
        Command::BenchInfer(args) => bench_infer(args),
        Command::BenchMtp(args) => bench_mtp(args),
        Command::BenchContext(args) => bench_context(args),
        Command::Status(args) => status(args),
    }
}

fn serve(args: ServeArgs) -> Result<()> {
    let config = if args.interactive {
        onboarding::server_config(&args)?
    } else if let Some(model) = args.model {
        ServerConfig {
            model,
            host: args.host,
            port: args.port,
            model_id: args.model_id,
        }
    } else if let Some(saved) = crate::state::load_server_config()? {
        saved
    } else {
        onboarding::server_config(&args)?
    };
    if args.dry_run {
        println!("{}", serde_json::to_string_pretty(&config)?);
        return Ok(());
    }
    crate::state::save_server_config(&config)?;
    crate::server::serve(config)
}

fn infer(args: InferArgs) -> Result<()> {
    let prompt = prompt_from_args(args.prompt, args.prompt_file)?;
    let request = InferenceRequest {
        model: args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string()),
        prompt,
        system: args.system,
        messages: Vec::new(),
        stop: args.stop,
        max_tokens: args.max_tokens,
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
        depth: args.depth,
        mtp: !args.no_mtp,
        requested_context_tokens: None,
        profile_timings: args.profile_timings,
    };
    let backend = crate::inference::NativeMlxBackend::new();
    let response = backend.infer(&request)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&response)?);
    } else {
        println!("{}", response.text);
    }
    Ok(())
}

fn status(args: StatusArgs) -> Result<()> {
    let saved = crate::state::load_server_config().ok().flatten();
    let payload = serde_json::json!({
        "version": env!("CARGO_PKG_VERSION"),
        "python_dependency": false,
        "state_path": crate::state::state_path(),
        "model_cache": crate::model::model_cache_dir(),
        "saved_server": saved,
        "native_backend": crate::inference::NativeMlxBackend::status(),
    });
    if args.json {
        println!("{}", serde_json::to_string_pretty(&payload)?);
    } else {
        println!("ferrite {}", env!("CARGO_PKG_VERSION"));
        println!("python dependency: no");
        println!("state: {}", crate::state::state_path().display());
        println!("model cache: {}", crate::model::model_cache_dir().display());
        println!(
            "native backend: {}",
            crate::inference::NativeMlxBackend::status()
        );
    }
    Ok(())
}

fn inspect(args: InspectArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let model = crate::model::LoadedModel::load(&model_ref)?;
    let summary = model.summary();
    if args.json {
        println!("{}", serde_json::to_string_pretty(&summary)?);
    } else {
        println!("model: {}", summary.path.display());
        println!(
            "architecture: {}",
            summary.architecture.as_deref().unwrap_or("unknown")
        );
        println!(
            "model_type: {}",
            summary.model_type.as_deref().unwrap_or("unknown")
        );
        println!(
            "hidden_size: {}",
            summary
                .hidden_size
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!(
            "layers: {}",
            summary
                .num_hidden_layers
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!(
            "vocab_size: {}",
            summary
                .vocab_size
                .map(|v| v.to_string())
                .unwrap_or_else(|| "unknown".to_string())
        );
        println!("model tensors: {}", summary.model_tensor_count);
        println!("mtp tensors: {}", summary.mtp_tensor_count);
        if let Ok(plan) = crate::qwen36::Qwen36Plan::from_model(&model) {
            println!(
                "linear-attention layers: {}",
                plan.linear_attention_layers.len()
            );
            println!(
                "full-attention layers: {}",
                plan.full_attention_layers.len()
            );
        }
    }
    Ok(())
}

fn bench_native(args: BenchNativeArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let result = crate::bench::run_native_microbench(
        &model_ref,
        &args.prompt,
        args.iterations,
        args.warmup,
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model: {}", result.model);
        println!("prompt tokens: {}", result.prompt_tokens);
        println!(
            "iterations: {} (+{} warmup)",
            result.iterations, result.warmup
        );
        println!("elapsed: {:.4}s", result.elapsed_s);
        println!("layer0 MLP passes/s: {:.3}", result.layer0_mlp_passes_per_s);
        println!(
            "prompt-token equivalent/s: {:.1}",
            result.prompt_token_equivalent_per_s
        );
        println!("output shape: {:?}", result.output_shape);
        println!(
            "first full-attention q/k/v: {:?} / {:?} / {:?}",
            result.first_full_attention_q_shape,
            result.first_full_attention_k_shape,
            result.first_full_attention_v_shape
        );
        println!(
            "first full-attention output (no RoPE/mask): {:?}",
            result.first_full_attention_output_shape
        );
        println!(
            "first full-attention block (no RoPE/mask): {:?}",
            result.first_full_attention_block_shape
        );
        println!(
            "first full-attention block (causal+RoPE): {:?}",
            result.first_full_attention_causal_rope_block_shape
        );
        println!(
            "first linear-attention q/k/v/a/b/z: {:?}",
            result.first_linear_attention_qkv_shapes
        );
        println!(
            "first linear-attention reference block: {:?}",
            result.first_linear_attention_reference_block_shape
        );
        println!(
            "ablated full-model logits (linear attention stubbed): {:?}",
            result.ablated_full_model_logits_shape
        );
        println!(
            "note: the full decode path uses cached decoding, custom Metal GDN kernels, and optional MTP speculative decoding"
        );
    }
    Ok(())
}

fn bench_decode(args: BenchDecodeArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = prompt_from_args(Some(args.prompt), args.prompt_file)?;
    let sampling = crate::sampling::SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    };
    let result = crate::bench::run_decode_bench(&model_ref, &prompt, args.tokens, sampling)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model: {}", result.model);
        println!("backend: {}", result.backend);
        println!(
            "sampling: temperature {:.3}, top_p {:.3}, top_k {}",
            result.temperature, result.top_p, result.top_k
        );
        println!("prompt tokens: {}", result.prompt_tokens);
        println!("generated tokens: {}", result.generated_tokens);
        println!("finish reason: {:?}", result.finish_reason);
        println!("load: {:.3}s", result.load_s);
        println!("prefill: {:.3}s", result.prefill_s);
        println!("decode: {:.3}s", result.decode_s);
        println!("total: {:.3}s", result.total_s);
        println!("output tok/s: {:.3}", result.tokens_per_s);
        if let Some(tps) = result.steady_state_tokens_per_s {
            println!("steady-state tok/s: {:.3}", tps);
        }
        println!("text: {:?}", result.text);
        println!(
            "note: tok/s excludes model load; steady-state excludes prefill and needs --tokens > 1"
        );
    }
    Ok(())
}

fn bench_infer(args: BenchInferArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let mut prompt = prompt_from_args(Some(args.prompt), args.prompt_file)?;
    if args.prompt_repeat > 1 {
        prompt = std::iter::repeat_n(prompt, args.prompt_repeat)
            .collect::<Vec<_>>()
            .join("\n");
    }
    let sampling = crate::sampling::SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    };
    let result = crate::bench::run_infer_path_bench(
        &model_ref,
        &prompt,
        args.tokens,
        sampling,
        args.mtp,
        args.depth,
        args.iterations,
        args.warmup,
        args.profile_timings,
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model: {}", result.model);
        println!("backend: {}", result.backend);
        println!("prompt tokens: {}", result.prompt_tokens);
        println!("requested tokens: {}", result.requested_tokens);
        println!(
            "iterations: {} (+{} warmup)",
            result.iterations, result.warmup
        );
        println!("load: {:.3}s", result.load_s);
        println!("avg prefill: {:.3}s", result.avg_prefill_s);
        println!("avg decode: {:.3}s", result.avg_decode_s);
        println!("avg total: {:.3}s", result.avg_total_s);
        println!("prefill tok/s: {:.1}", result.prefill_tokens_per_s);
        println!("output tok/s: {:.3}", result.output_tokens_per_s);
        println!("completion tokens: {}", result.completion_tokens);
    }
    Ok(())
}

fn bench_mtp(args: BenchMtpArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = prompt_from_args(Some(args.prompt), args.prompt_file)?;
    let result = crate::bench::run_mtp_bench(&model_ref, &prompt, args.tokens, args.depth)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model: {}", result.model);
        println!("backend: {}", result.backend);
        println!("prompt tokens: {}", result.prompt_tokens);
        println!("requested tokens: {}", result.requested_tokens);
        println!("generated tokens: {}", result.generated_tokens);
        println!("depth: {}", result.depth);
        println!("finish reason: {:?}", result.finish_reason);
        println!("load: {:.3}s", result.load_s);
        println!("prefill: {:.3}s", result.prefill_s);
        println!("decode: {:.3}s", result.decode_s);
        println!("total: {:.3}s", result.total_s);
        println!("output tok/s: {:.3}", result.tokens_per_s);
        println!("draft tokens: {}", result.draft_tokens);
        println!("accepted drafts: {}", result.accepted_draft_tokens);
        println!("rejected drafts: {}", result.rejected_draft_tokens);
        println!("acceptance rate: {:.1}%", result.acceptance_rate * 100.0);
        println!("text: {:?}", result.text);
        println!(
            "note: bench-mtp uses greedy verified speculative decoding for acceptance tracking"
        );
    }
    Ok(())
}

fn bench_context(args: BenchContextArgs) -> Result<()> {
    let model_ref = args.model.unwrap_or_else(|| DEFAULT_MODEL.to_string());
    let prompt = prompt_from_args(Some(args.prompt), args.prompt_file)?;
    let contexts = parse_contexts(&args.contexts)?;
    let sampling = crate::sampling::SamplingConfig {
        temperature: args.temperature,
        top_p: args.top_p,
        top_k: args.top_k,
    };
    let result = crate::bench::run_context_probe(
        &model_ref,
        &prompt,
        &contexts,
        args.generate_tokens,
        sampling,
    )?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&result)?);
    } else {
        println!("model: {}", result.model);
        println!("backend: {}", result.backend);
        println!("source prompt tokens: {}", result.source_prompt_tokens);
        println!("generated tokens per case: {}", result.generate_tokens);
        for case in result.contexts {
            println!(
                "{} prompt tokens: prefill {:.3}s ({:.1} tok/s), decode {:.3}s ({:.3} tok/s), est KV {:.2} GiB, est prompt hidden {:.2} GiB",
                case.prompt_tokens,
                case.prefill_s,
                case.prefill_tokens_per_s,
                case.decode_s,
                case.output_tokens_per_s,
                case.estimated_full_attention_kv_gib,
                case.estimated_prompt_hidden_gib,
            );
        }
        println!(
            "note: prompt tokens are synthetic repetitions of the prompt; this measures runtime capacity, not prompt quality"
        );
    }
    Ok(())
}

fn parse_contexts(value: &str) -> Result<Vec<usize>> {
    let mut contexts = Vec::new();
    for part in value.split(',') {
        let trimmed = part.trim();
        if trimmed.is_empty() {
            continue;
        }
        let parsed = trimmed
            .parse::<usize>()
            .with_context(|| format!("parse context token count {trimmed:?}"))?;
        contexts.push(parsed);
    }
    if contexts.is_empty() {
        anyhow::bail!("at least one context token count is required");
    }
    Ok(contexts)
}

fn prompt_from_args(prompt: Option<String>, prompt_file: Option<PathBuf>) -> Result<String> {
    if let Some(path) = prompt_file {
        return std::fs::read_to_string(&path)
            .with_context(|| format!("read prompt file {}", path.display()));
    }
    match prompt {
        Some(prompt) => Ok(prompt),
        None => onboarding::ask_prompt_text(),
    }
}

impl Default for ServeArgs {
    fn default() -> Self {
        Self {
            model: None,
            host: "127.0.0.1".to_string(),
            port: 11434,
            model_id: DEFAULT_PUBLIC_MODEL_ID.to_string(),
            interactive: false,
            dry_run: false,
        }
    }
}
