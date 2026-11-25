#!/usr/bin/env python3
"""
vLLM OpenAI-Compatible Server for Local Model Inference

Usage:
    python vllm_server.py --model /data1/test/models/Qwen3-32B --port 8000
    python vllm_server.py --model /data1/test/models/Qwen3-32B --port 8000 --tensor-parallel-size 2
"""

import os
import sys
import argparse
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 默认配置
DEFAULT_MODEL_PATH = "/data1/test/models/Qwen3-32B"
DEFAULT_PORT = 8022
DEFAULT_HOST = "0.0.0.0"


def parse_args():
    parser = argparse.ArgumentParser(description="vLLM OpenAI-Compatible Server")
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                        help='Path to the model directory')
    parser.add_argument('--served-model-name', type=str, default=None,
                        help='Model name exposed via API (default: model directory name)')
    parser.add_argument('--host', type=str, default=DEFAULT_HOST,
                        help='Host to bind the server')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT,
                        help='Port to bind the server')
    parser.add_argument('--tensor-parallel-size', type=int, default=1,
                        help='Number of GPUs for tensor parallelism')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.5,
                        help='GPU memory utilization (0.0 to 1.0)')
    parser.add_argument('--max-model-len', type=int, default=None,
                        help='Maximum model context length (auto-detected if not set)')
    parser.add_argument('--dtype', type=str, default='bfloat16',
                        choices=['auto', 'float16', 'bfloat16', 'float32'],
                        help='Data type for model weights')
    parser.add_argument('--max-num-seqs', type=int, default=16,
                        help='Maximum number of sequences per iteration')
    parser.add_argument('--enable-prefix-caching', action='store_true',
                        help='Enable automatic prefix caching')
    parser.add_argument('--api-key', type=str, default=None,
                        help='API key for authentication (optional)')
    return parser.parse_args()


def check_vllm_installed():
    """检查 vLLM 是否已安装"""
    try:
        import vllm
        logger.info(f"vLLM version: {vllm.__version__}")
        return True
    except ImportError:
        logger.error("vLLM is not installed. Please install it with: pip install vllm")
        return False


def start_server(args):
    """启动 vLLM OpenAI 兼容服务"""
    
    if not check_vllm_installed():
        sys.exit(1)
    
    # 检查模型路径
    if not os.path.exists(args.model):
        logger.error(f"Model path does not exist: {args.model}")
        sys.exit(1)
    
    # 设置模型名称
    served_name = args.served_model_name or os.path.basename(args.model.rstrip('/'))
    
    # 构建命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--served-model-name", served_name,
        "--host", args.host,
        "--port", str(args.port),
        "--tensor-parallel-size", str(args.tensor_parallel_size),
        "--gpu-memory-utilization", str(args.gpu_memory_utilization),
        "--dtype", args.dtype,
        "--max-num-seqs", str(args.max_num_seqs),
        "--trust-remote-code",
    ]
    
    if args.max_model_len:
        cmd.extend(["--max-model-len", str(args.max_model_len)])
    
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    
    if args.api_key:
        cmd.extend(["--api-key", args.api_key])
    
    # 打印启动信息
    logger.info("=" * 60)
    logger.info("vLLM OpenAI-Compatible Server")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Served Name: {served_name}")
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Tensor Parallel Size: {args.tensor_parallel_size}")
    logger.info(f"GPU Memory Utilization: {args.gpu_memory_utilization}")
    logger.info(f"Data Type: {args.dtype}")
    logger.info("=" * 60)
    logger.info(f"API Base URL: http://{args.host}:{args.port}/v1")
    logger.info("=" * 60)
    logger.info("")
    logger.info("To use with eval_bc.py, set environment variables:")
    logger.info(f"  export OPENAI_BASE_URL=http://localhost:{args.port}/v1")
    logger.info("  export OPENAI_API_KEY=dummy")
    logger.info(f"  python scripts/eval_bc.py --model_name {served_name} --tokenizer_path {args.model} ...")
    logger.info("")
    logger.info("Starting server...")
    logger.info("")
    
    # 启动服务
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except subprocess.CalledProcessError as e:
        logger.error(f"Server failed with exit code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    args = parse_args()
    start_server(args)
