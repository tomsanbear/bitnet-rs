build:
	cargo build

tests = test-cpu test-metal test-cuda test-flash-attn

test-cpu:
	cargo test

test-metal:
	cargo test --features "metal"

test-cuda:
	cargo test --features "cuda"

test-flash-attn:
	cargo test --features "flash_attn"

.PHONY: $(tests)
test-all: $(tests)