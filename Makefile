SHELL := /bin/zsh

RUSTFLAGS_NATIVE := -C target-cpu=native
PREFIX ?= $(HOME)/.local
BINDIR ?= $(PREFIX)/bin

.PHONY: check test build build-native install inspect bench-native bench-decode bench-decode-json bench-mtp bench-mtp-json bench-context bench-context-json bench-depth-greedy-json bench-depth-json bench-flappy-ar-json bench-flappy-mtp-json bench-coding-tail-ar-json bench-coding-tail-mtp-json bench-fuse-gdn bench-fuse-mlp run-server clean

check:
	cargo check

test:
	cargo test

build:
	cargo build --release

build-native:
	RUSTFLAGS="$(RUSTFLAGS_NATIVE)" cargo build --release

install: build
	install -d "$(BINDIR)"
	install -m 0755 target/release/ferrite "$(BINDIR)/ferrite"

inspect:
	cargo run -- inspect

bench-native:
	cargo run --release -- bench-native --iterations 10 --warmup 2

bench-decode:
	cargo run --release -- bench-decode --tokens 4

bench-decode-json:
	cargo run --release -- bench-decode --tokens 16 --json

bench-mtp:
	cargo run --release -- bench-mtp --tokens 16 --depth 2

bench-mtp-json:
	cargo run --release -- bench-mtp --tokens 16 --depth 2 --json

bench-context:
	cargo run --release -- bench-context --contexts 8192,16384,32768 --generate-tokens 4

bench-context-json:
	cargo run --release -- bench-context --contexts 8192,16384,32768 --generate-tokens 4 --json

bench-depth-greedy-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'for depth in 1 2 3 4 5; do cargo run --release -- infer --prompt-file examples/prompts/flappy.txt --max-tokens 128 --depth $$depth --temperature 0 --top-k 1 --json; done'

bench-depth-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'for depth in 1 2 3 4 5; do cargo run --release -- infer --prompt-file examples/prompts/flappy.txt --max-tokens 128 --depth $$depth --json; done'

bench-flappy-ar-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'cargo run --release -- bench-decode --prompt-file examples/prompts/flappy.txt --tokens 64 --temperature 0 --top-k 1 --json'

bench-flappy-mtp-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'cargo run --release -- infer --prompt-file examples/prompts/flappy.txt --max-tokens 64 --depth 2 --temperature 0 --top-k 1 --json'

bench-coding-tail-ar-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'cargo run --release -- bench-decode --prompt-file examples/prompts/coding-agent-tail.txt --tokens 64 --temperature 0 --top-k 1 --json'

bench-coding-tail-mtp-json:
	env -u MAKELEVEL -u MAKEFLAGS -u MFLAGS /bin/zsh -lc 'cargo run --release -- infer --prompt-file examples/prompts/coding-agent-tail.txt --max-tokens 64 --depth 2 --temperature 0 --top-k 1 --json'

bench-fuse-gdn:
	MTPLX_FUSE_GDN_PROJECTIONS=1 cargo run --release -- bench-decode --tokens 16 --json

bench-fuse-mlp:
	MTPLX_FUSE_MLP_PROJECTIONS=1 cargo run --release -- bench-decode --tokens 16 --json

run-server:
	cargo run --release -- serve --interactive

clean:
	cargo clean
