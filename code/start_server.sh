#!/bin/bash
# Dedicated, reproducible Ollama instance for benchmarking (separate from the shared server on :11434).
export OLLAMA_HOST=127.0.0.1:11500
export OLLAMA_MODELS=/mnt/data/private_ai/ollama
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_NOPRUNE=1
exec ollama serve
