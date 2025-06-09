# Brush Makefile
# Provides convenient commands for building and running Brush

.PHONY: help run run-clean run-viewer clean build test

# Default target
help:
	@echo "Brush Build Commands:"
	@echo "  make run         - Run brush_app normally"
	@echo "  make run-clean   - Run brush_app with cleared Homebrew paths (macOS Sequoia fix)"
	@echo "  make run-viewer  - Run brush_app with --with-viewer flag"
	@echo "  make build       - Build the application"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make test        - Run tests"

# Standard run
run:
	cargo run --bin brush_app

# Run with cleared Homebrew library paths (macOS Sequoia bus error fix)
run-clean:
ifeq ($(shell uname),Darwin)
	@echo "Running with cleared Homebrew library paths (macOS Sequoia fix)..."
	env DYLD_LIBRARY_PATH="" DYLD_FALLBACK_LIBRARY_PATH="" cargo run --bin brush_app
else
	cargo run --bin brush_app
endif

# Run with viewer
run-viewer:
ifeq ($(shell uname),Darwin)
	@echo "Running with viewer and cleared Homebrew library paths (macOS Sequoia fix)..."
	env DYLD_LIBRARY_PATH="" DYLD_FALLBACK_LIBRARY_PATH="" cargo run --bin brush_app -- --with-viewer
else
	cargo run --bin brush_app -- --with-viewer
endif

# Build
build:
	cargo build --bin brush_app

# Clean
clean:
	cargo clean

# Test
test:
	cargo test
