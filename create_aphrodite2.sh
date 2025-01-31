#!/bin/bash
MODEL_NAME="erax-ai/EraX-VL-7B-V1.5"
CONTAINER_NAME="aphrodite_${MODEL_NAME//[^a-zA-Z0-9_.-]/_}_0.4GPU"

docker run --runtime nvidia --gpus all -d \
    --name "${CONTAINER_NAME}" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env CUDA_VISIBLE_DEVICES=0,1,2,3 \
    --env APHRODITE_IMAGE_FETCH_TIMEOUT=60 \
    -p 2242:2242 \
    --ipc=host \
    my-aphrodite-openai:cuda12.1 \
    --model "${MODEL_NAME}" \
    --dtype float16 \
    --rope-scaling '{"type":"linear","factor":1.0,"mrope_section":[16,24,24],"rope_type":"default"}' \
    --gpu-memory-utilization 0.5 \
    --tensor-parallel-size 4 \
    --disable-frontend-multiprocessing



# docker run --runtime nvidia --gpus all -d \
#     --name ${CONTAINER_NAME} \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=0,1" \
#     --env "APHRODITE_IMAGE_FETCH_TIMEOUT=60" \
#     -p 2244:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model 5CD-AI/Vintern-1B-v3_5 \
#     --dtype float16 \
#     --gpu-memory-utilization 0.4 \
#     --tensor-parallel-size 2 \
#     --disable-frontend-multiprocessing


    # --enforce-eager \
    # --enable-chunked-prefill \
# #!/bin/bash
# docker run --runtime nvidia --gpus all -d \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=0" \
#     -p 2242:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-7B-V1.5 \
#     --enforce-eager \
#     --dtype half \
#     --enable-chunked-prefill \
#     --kv-cache-dtype fp8_e4m3 \
#     --max-model-len 8192 \
#     --rope-scaling "{\"type\":\"linear\",\"factor\":1.0,\"mrope_section\":[16,24,24],\"rope_type\":\"default\"}" \
#     --gpu-memory-utilization 0.85 \
#     --max-num-batched-tokens 4096 \
#     --block-size 8 \
#     --tensor-parallel-size 1 \
#     --scheduler-delay-factor 0.5 \
#     --use-v2-block-manager
# docker run --runtime nvidia --gpus all -d \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=0" \
#     -p 2242:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-7B-V1.5 \
#     --enforce-eager \
#     --dtype half \
#     --rope-scaling "{\"type\":\"linear\",\"factor\":1.0,\"mrope_section\":[16,24,24],\"rope_type\":\"default\"}" \
#     --gpu-memory-utilization 0.905

#!/bin/bash 

# Set environment variables for GPU communication
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=0
# export NCCL_SOCKET_IFNAME=eth0

# # Run Aphrodite with tensor parallelism
# docker run --runtime nvidia --gpus all -d \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=1,0" \
#     --env "NCCL_DEBUG=INFO" \
#     --env "NCCL_IB_DISABLE=0" \
#     --env "NCCL_SOCKET_IFNAME=eth0" \
#     -p 2242:2242 \
#     --ipc=host \
#     --privileged \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-7B-V1.5 \
#     --tensor-parallel-size 2 \
#     --dtype half \
#     --enforce-eager \
#     --rope-scaling "{\"type\":\"linear\",\"factor\":1.0,\"mrope_section\":[16,24,24],\"rope_type\":\"default\"}" \
#     --gpu-memory-utilization 0.95



# docker run --runtime nvidia --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=0,1,2" \
#     -p 2243:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-2B-V1.5 \
#     --tensor-parallel-size 1 \
#     --api-keys "sk-empty" \
#     --enforce-eager



# docker run --runtime nvidia --gpus all -d \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=1,2" \
#     -p 2242:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-7B-V1.5 \
#     --tensor-parallel-size 1 \
#     --enforce-eager \
#     --rope-scaling "{\"type\":\"dynamic\",\"factor\":1.0}"


# # Build and run commands
# docker build -t my-aphrodite-openai:cuda12.1 .

# docker run --runtime nvidia --gpus all \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     --env "CUDA_VISIBLE_DEVICES=0,1,2" \
#     -p 2242:2242 \
#     --ipc=host \
#     my-aphrodite-openai:cuda12.1 \
#     --model erax-ai/EraX-VL-2B-V1.5 \
#     --tensor-parallel-size 1 \
#     --api-keys "sk-empty"