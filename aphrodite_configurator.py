import subprocess
import json
import os
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class GPUInfo:
    name: str
    memory: float
    compute_cap: str
    cuda_version: Optional[str] = None

class AphroditeConfigurator:
    def __init__(self):
        self.gpu_info = self._get_gpu_info()
        self.base_config = {
            "model": "erax-ai/EraX-VL-7B-V1.5",
            "host": "0.0.0.0",
            "port": 2242,
            "device": "cuda"
        }
        self.model_config = {
            "dtype": "half",
            "enforce_eager": True,
            "trust_remote_code": True,
            "max_model_len": 8192
        }
        self.cache_config = {
            "gpu_memory_utilization": 0.85,
            "block_size": 8,
            "kv_cache_dtype": "fp8_e4m3",
            "enable_prefix_caching": True
        }
        self.scheduler_config = {
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "scheduler_delay_factor": 0.5,
            "use_v2_block_manager": True
        }
        self.parallel_config = {
            "tensor_parallel_size": 1
        }

    def _get_gpu_info(self) -> Optional[GPUInfo]:
        try:
            # Get GPU info
            nvidia_smi = subprocess.check_output([
                "nvidia-smi", 
                "--query-gpu=gpu_name,memory.total,compute_cap,driver_version",
                "--format=csv,noheader,nounits"
            ]).decode('utf-8').strip().split('\n')[0].split(', ')
            
            return GPUInfo(
                name=nvidia_smi[0],
                memory=float(nvidia_smi[1]),
                compute_cap=nvidia_smi[2],
                cuda_version=nvidia_smi[3]
            )
        except Exception as e:
            print(f"Error getting GPU info: {e}")
            return None

    def _detect_primary_gpu(self) -> str:
        """Detect the index of the primary GPU."""
        try:
            output = subprocess.check_output([
                "nvidia-smi", "--query-gpu=index", "--format=csv,noheader"
            ]).decode('utf-8').strip()
            return output.split("\n")[0]  # Return the index of the first GPU
        except Exception as e:
            print(f"Unable to detect primary GPU: {e}")
            return "0"  # Default to GPU 0 if detection fails

    def optimize_for_pascal(self):
        if self.gpu_info and "P40" in self.gpu_info.name:
            self.cache_config.update({
                "kv_cache_dtype": "fp8_e4m3",
                "block_size": 8,
                "gpu_memory_utilization": 0.85
            })
            self.model_config.update({
                "dtype": "half",
                "max_model_len": 8192
            })
            self.scheduler_config.update({
                "enable_chunked_prefill": True,
                "max_num_batched_tokens": 4096
            })
            self.parallel_config.update({
                "tensor_parallel_size": 1
            })

    def generate_config(self):
        self.optimize_for_pascal()

        gpu_index = self._detect_primary_gpu()
        config = {
            **self.base_config,
            **self.model_config, 
            **self.cache_config,
            **self.scheduler_config,
            **self.parallel_config,
            "rope_scaling": {
                "type": "linear",
                "factor": 1.0,
                "mrope_section": [16, 24, 24],
                "rope_type": "default"
            },
            "gpu_index": gpu_index
        }
        
        return self.create_launch_script(config)

    def create_launch_script(self, config: Dict[str, Any]) -> str:
        script = """#!/bin/bash
docker run --runtime nvidia --gpus all -d \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    --env "CUDA_VISIBLE_DEVICES={gpu_index}" \
    -p {port}:{port} \
    --ipc=host \
    my-aphrodite-openai:cuda12.1 \
    --model {model} \
    --enforce-eager \
    --dtype {dtype} \
    --enable-chunked-prefill \
    --kv-cache-dtype {kv_cache_dtype} \
    --max-model-len {max_model_len} \
    --rope-scaling '{rope_scaling}' \
    --gpu-memory-utilization {gpu_memory_utilization} \
    --max-num-batched-tokens {max_num_batched_tokens} \
    --block-size {block_size} \
    --tensor-parallel-size {tensor_parallel_size} \
    --scheduler-delay-factor {scheduler_delay_factor} \
    --use-v2-block-manager""".format(
            **config,
            rope_scaling=json.dumps(config["rope_scaling"])
        )
        
        script_path = "create_aphrodite.sh"
        with open(script_path, "w") as f:
            f.write(script)
        os.chmod(script_path, 0o755)
        return script_path

if __name__ == "__main__":
    configurator = AphroditeConfigurator()
    script_path = configurator.generate_config()
    if configurator.gpu_info:
        print(f"Created optimized configuration for {configurator.gpu_info.name} at: {script_path}")
    else:
        print("Failed to detect GPU info. Configuration may not be optimal.")
