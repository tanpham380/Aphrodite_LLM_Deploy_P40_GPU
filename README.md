# Aphrodite LLM Deployment for Pascal GPUs

## Overview
This project provides customized configurations and Docker setup for running modern Large Language Models (LLMs) on legacy NVIDIA Pascal GPUs like the Tesla P40. It uses a modified version of the Aphrodite Engine with CUDA 12.1 compatibility.

**Ngôn ngữ**: [English](#overview) | [Tiếng Việt](#aphrodite-trien-khai-mo-hinh-llm-cho-gpu-pascal)

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

---

# Aphrodite Triển khai Mô hình LLM cho GPU Pascal

## Tổng quan
Dự án này cung cấp cấu hình tùy chỉnh và thiết lập Docker để chạy các Mô hình Ngôn ngữ Lớn (LLMs) hiện đại trên các GPU NVIDIA Pascal cũ như Tesla P40. Nó sử dụng phiên bản đã chỉnh sửa của Aphrodite Engine tương thích với CUDA 12.1.

## Tính năng
- **Tối ưu hóa cho kiến trúc GPU Pascal của NVIDIA**
- **Cấu hình Docker tùy chỉnh với hỗ trợ CUDA 12.1**
- **Cài đặt tiết kiệm bộ nhớ cho các GPU cũ**
- **Tự động cấu hình qua `aphrodite_configurator.py`**
- **Hỗ trợ các định dạng mô hình và lượng tử hóa khác nhau**

## Yêu cầu
- **GPU NVIDIA** với kiến trúc Pascal (GTX 1000 series, Tesla P40, v.v.)
- **Docker** với NVIDIA Container Runtime
- **Driver NVIDIA** hỗ trợ CUDA 12.1
- **Ít nhất 16GB VRAM GPU** (khuyến nghị)

## Bắt đầu nhanh

### Xây dựng Docker image
```bash
docker build -t my-aphrodite-openai:cuda12.1 .
```

### Chạy mô hình
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

### Ví dụ: create_aphrodite2.sh
Dưới đây là một script minh họa để tự động quá trình tạo:

```bash
#!/bin/bash

# Xây dựng Docker image
docker build -t my-aphrodite-openai:cuda12.1 .

# Chạy Docker container
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

## Cấu hình
Script `aphrodite_configurator.py` cung cấp tối ưu hóa tự động cho các GPU Pascal với các cài đặt như:

- **Bộ nhớ đệm KV 8-bit (FP8 E4M3)**
- **Chunked prefill**
- **Kiểm soát sử dụng bộ nhớ**
- **Dynamic rope scaling**
- **Tối ưu hóa kích thước block**

### Cài đặt khuyến nghị
Dành cho GPU Pascal, chúng tôi khuyến nghị:
- Sử dụng `--gpu-memory-utilization` từ **0.8-0.9**
- Kích hoạt bộ nhớ đệm KV 8-bit với `--kv-cache-dtype fp8_e4m3`
- Cài đặt `--block-size` phù hợp (mặc định: 8)

## Quản lý bộ nhớ
Quản lý bộ nhớ hiệu quả rất quan trọng khi chạy LLMs trên GPU Pascal. Tuân theo các hướng dẫn sau:
- Đảm bảo kích hoạt **bộ nhớ đệm KV 8-bit** để giảm footprint bộ nhớ.
- Sử dụng **chunked prefill** để tối ưu hóa phân bổ bộ nhớ.
- Điều chỉnh `--gpu-memory-utilization` để tránh quá tải bộ nhớ GPU.

## Lời cảm ơn
Dự án này dựa trên Aphrodite Engine với các tùy chỉnh dành cho tương thích GPU Pascal.

## Giấy phép
Dự án này tuân theo giấy phép giống như Aphrodite Engine. Tham khảo tài liệu giấy phép để biết thêm chi tiết.
