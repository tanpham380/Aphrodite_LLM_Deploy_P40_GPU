# Aphrodite LLM Deployment for Pascal GPUs

## Overview
This project provides customized configurations and Docker setup for running modern Large Language Models (LLMs) on legacy NVIDIA Pascal GPUs like the Tesla P40. It uses a modified version of the Aphrodite Engine with CUDA 12.1 compatibility.

## Features
- **Optimized for NVIDIA Pascal architecture GPUs**
- **Custom Docker configuration with CUDA 12.1 support**
- **Memory-efficient settings for older GPUs**
- **Automated configuration through `aphrodite_configurator.py`**
- **Support for various model formats and quantization**

## Requirements
- **NVIDIA GPU** with Pascal architecture (GTX 1000 series, Tesla P40, etc.)
- **Docker** with NVIDIA Container Runtime
- **NVIDIA Driver** supporting CUDA 12.1
- **At least 16GB GPU VRAM** (recommended)

## Quick Start

### Build the Docker image
```bash
docker build -t my-aphrodite-openai:cuda12.1 .
```

### Run the model
```bash
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "CUDA_VISIBLE_DEVICES=0" \
    -p 2242:2242 \
    --ipc=host \
    my-aphrodite-openai:cuda12.1 \
    --model erax-ai/EraX-VL-7B-V1.5 \
    --enforce-eager \
    --dtype half \
    --gpu-memory-utilization 0.85
```

### Example: create_aphrodite2.sh
Below is an example script to automate the creation process:

```bash
#!/bin/bash

# Build Docker image
docker build -t my-aphrodite-openai:cuda12.1 .

# Run Docker container
docker run --runtime nvidia --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "CUDA_VISIBLE_DEVICES=0" \
    -p 2242:2242 \
    --ipc=host \
    my-aphrodite-openai:cuda12.1 \
    --model erax-ai/EraX-VL-7B-V1.5 \
    --enforce-eager \
    --dtype half \
    --gpu-memory-utilization 0.85
```

## Configuration
The `aphrodite_configurator.py` script provides automatic optimization for Pascal GPUs with settings like:

- **8-bit KV cache (FP8 E4M3)**
- **Chunked prefill**
- **Memory utilization control**
- **Dynamic rope scaling**
- **Block size optimization**

### Recommended Settings
For Pascal GPUs, we recommend:
- Using `--gpu-memory-utilization` between **0.8-0.9**
- Enabling 8-bit KV cache with `--kv-cache-dtype fp8_e4m3`
- Setting appropriate `--block-size` (default: 8)

## Memory Management
Efficient memory management is critical for running LLMs on Pascal GPUs. Follow these guidelines:
- Ensure **8-bit KV cache** is enabled for reduced memory footprint.
- Use **chunked prefill** to optimize memory allocation.
- Adjust `--gpu-memory-utilization` to prevent overloading the GPU memory.

## Acknowledgments
This project is based on the Aphrodite Engine with customizations for Pascal GPU compatibility.

## License
This project adheres to the same license as the Aphrodite Engine. Refer to the license documentation for details.
