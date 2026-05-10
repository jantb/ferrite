# Ferrite

Native MTP inference for Apple Silicon agents.

Ferrite is a Rust inference server for local Qwen3.6 MTPLX-style models on Apple Silicon. It runs without Python, exposes OpenAI-compatible endpoints, defaults to Ollama's local port, and keeps setup intentionally small: pick a model once, then reuse the saved config.

## Status

Ferrite is early and focused on one path:

- Native Rust + MLX backend
- Qwen3.6 27B MTPLX optimized model support
- MTP speculative decoding
- OpenAI-compatible `/v1/chat/completions`, `/v1/completions`, `/v1/models`
- SSE streaming for requests without stop sequences
- Warm prefix cache for agent-style repeated context
- Context and decode benchmarks

## Requirements

- macOS on Apple Silicon
- Rust toolchain
- Local model files with `config.json`, `tokenizer.json`, and safetensors

By default Ferrite looks for models under:

```sh
~/.mtplx/models
```

Override that with:

```sh
export FERRITE_MODEL_DIR=/path/to/models
```

The current verified default model is:

```text
Youssofal/Qwen3.6-27B-MTPLX-Optimized-Speed
```

## Install

Build:

```sh
make build
```

Install to `~/.local/bin/ferrite`:

```sh
make install
```

Install somewhere else:

```sh
make install PREFIX=/usr/local
```

## First Run

Start Ferrite:

```sh
ferrite
```

On first run, Ferrite asks which model to use and saves the selection. Later runs reuse the saved config automatically.

The saved config is stored at:

```sh
~/.ferrite/server.json
```

Force setup again:

```sh
ferrite serve --interactive
```

Ferrite always binds to localhost by default:

```text
127.0.0.1:11434
```

## API

List models:

```sh
curl http://127.0.0.1:11434/v1/models
```

Chat completion:

```sh
curl http://127.0.0.1:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "ferrite-qwen36-27b-optimized-speed",
    "messages": [{"role": "user", "content": "Say hi"}],
    "max_tokens": 32,
    "stream": true
  }'
```

The old model alias `mtplx-qwen36-27b-optimized-speed` is accepted for compatibility.

## CLI

Run one local inference:

```sh
ferrite infer "Write a small Rust function that adds two numbers" --max-tokens 128
```

JSON output with timing and MTP stats:

```sh
ferrite infer "hi" --max-tokens 16 --json
```

Use autoregressive decoding without MTP:

```sh
ferrite infer "hi" --max-tokens 16 --no-mtp
```

## Benchmarks

Context probe:

```sh
make bench-context-json
```

Custom context sweep:

```sh
ferrite bench-context --contexts 8192,16384,32768,65536 --generate-tokens 4 --json
```

Decode benchmark:

```sh
make bench-decode-json
```

Server inference-path benchmark:

```sh
ferrite bench-infer --prompt "hi" --prompt-repeat 1000 --tokens 1 --json
```

MTP benchmark:

```sh
make bench-mtp-json
```

## Agent Use

Ferrite also implements the Ollama endpoints used by the local `agent` client:

```sh
agent --ollama-url http://127.0.0.1:11434 --model ferrite-qwen36-27b-optimized-speed
```

Point OpenAI-compatible clients at:

```text
http://127.0.0.1:11434/v1
```

Use model:

```text
ferrite-qwen36-27b-optimized-speed
```

Ferrite can keep a small warm prefix cache in-process. It is disabled by default to keep long-running agent sessions from retaining KV memory between turns.

Useful cache environment variables:

```sh
FERRITE_PREFIX_CACHE=0
FERRITE_PREFIX_CACHE_ENTRIES=1
FERRITE_PREFIX_CACHE_MAX_TOKENS=16384
FERRITE_PREFIX_CACHE_MAX_BYTES=1073741824
FERRITE_MAX_KV_CONTEXT_TOKENS=16384
FERRITE_PREFILL_CHUNK_TOKENS=128
FERRITE_PREFILL_EVAL_INTERVAL_CHUNKS=1
FERRITE_FULL_KV_CACHE_STEP=256
FERRITE_FULL_KV_CACHE_APPEND=tail_owned
FERRITE_FULL_KV_CACHE_PREFILL_RESERVE=1
FERRITE_DECODE_BLOCK_MASK=cpu
FERRITE_SMALL_M_QMV4=1
FERRITE_SMALL_M_QMV4_M_VALUES=1,4,5,6
FERRITE_SMALL_M_QMV4_SIMDGROUPS=4
FERRITE_SMALL_M_QMV4_STRICT=0
FERRITE_SPLIT_FULL_ATTN=1
FERRITE_SPLIT_FULL_ATTN_THRESHOLD=1024
FERRITE_SPLIT_FULL_ATTN_CHUNK_TOKENS=128
FERRITE_BLOCKWISE_FULL_ATTN=0
FERRITE_BLOCKWISE_FULL_ATTN_THRESHOLD=1024
FERRITE_BLOCKWISE_FULL_ATTN_BLOCK_TOKENS=512
FERRITE_PREFILL_EVAL_LAYER_INTERVAL=0
FERRITE_MLX_MEMORY_LIMIT_BYTES=8589934592
FERRITE_MLX_CACHE_LIMIT_BYTES=536870912
FERRITE_MEMORY_WATCHDOG=1
FERRITE_MEMORY_WATCHDOG_INTERVAL_MS=250
FERRITE_KILL_SWITCH_FILE=ferrite-kill-switch
FERRITE_MEMORY_LOG=ferrite-memory.log
FERRITE_MEMORY_TRACE=1
FERRITE_POST_GENERATION_CACHE=0
FERRITE_CHAT_POST_GENERATION_CACHE=0
```

Create `ferrite-kill-switch` in the working directory to make Ferrite reject the next checked inference phase. During an active request the memory watchdog treats the same file as a hard process kill, so it can interrupt long MLX calls that do not return to Rust quickly.
`FERRITE_RSS_KILL_BYTES` defaults to 50% of physical memory when unset; set it lower for a stricter process kill, or `0` to disable the RSS limit.
`FERRITE_MLX_ACTIVE_KILL_BYTES` defaults to 80% of physical memory when unset; set it lower to stop MLX active-memory spikes earlier, or `0` to disable the active-memory limit.
Prompt prefill is evaluated in chunks by default so tool-heavy requests do not build one large MLX graph before generation starts. Full-attention K/V cache storage grows in `FERRITE_FULL_KV_CACHE_STEP` token blocks to avoid concatenating and copying the entire dense cache on every prefill chunk. `FERRITE_FULL_KV_CACHE_PREFILL_RESERVE=1` reserves one tail-owned growth slot during prefill so the first decode token does not immediately reallocate and copy the prompt cache. `FERRITE_SPLIT_FULL_ATTN_*`, `FERRITE_BLOCKWISE_FULL_ATTN`, `FERRITE_DECODE_BLOCK_MASK=mlx`, `FERRITE_PREFILL_EVAL_INTERVAL_CHUNKS`, `FERRITE_PREFILL_EVAL_LAYER_INTERVAL`, and `FERRITE_FULL_KV_CACHE_APPEND=concat` are diagnostic tuning knobs; the defaults are the measured safe path.
`FERRITE_SMALL_M_QMV4=1` enables Ferrite's Rust-owned qmv4 path for small-M quantized linears. It defaults to `FERRITE_SMALL_M_QMV4_M_VALUES=1,4,5,6`, based on the measured fast cases on this M4 machine; use `FERRITE_SMALL_M_QMV4_M_VALUES=all` or the legacy `FERRITE_SMALL_M_QMV4_MAX_M=6` to force every M=1..6 case through the Ferrite kernel. `FERRITE_SMALL_M_QMV4_SIMDGROUPS` can be tuned with `bench-qmv4`; valid values are `1`, `2`, `4`, and `8`. `FERRITE_SMALL_M_QMV4_STRICT=1` turns fallback shader failures into request errors.

## Notes

- Streaming with stop sequences currently falls back to a conservative full-response path.
- Long-context support is model-capable, but practical limits depend on memory pressure and prefill cost.
- The project still uses `.mtplx` as the default model cache path for compatibility with existing local models.
