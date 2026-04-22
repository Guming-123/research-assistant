#!/usr/bin/env python3
"""
GPU Setup Script for Conda Environment
自动配置 GPU 环境
"""

import subprocess
import sys
import os

def run_command(cmd, description=""):
    """运行命令并显示输出"""
    if description:
        print(f"\n[{description}]")
        print(f"Running: {cmd}")

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=False,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=" * 50)
    print("GPU Environment Setup for Conda")
    print("=" * 50)
    print()

    # 1. 激活 conda 环境
    print("[1/6] Activating research_assistant environment...")
    if not run_command("conda activate research_assistant", "Activate conda"):
        print("\nEnvironment not found. Creating...")
        run_command("conda create -n research_assistant python=3.10 -y", "Create environment")
        run_command("conda activate research_assistant", "Activate conda")

    # 2. 检查 CUDA
    print("\n[2/6] Checking CUDA version...")
    run_command("nvidia-smi", "Check NVIDIA GPU")

    # 3. 安装 PyTorch
    print("\n[3/6] Installing PyTorch with CUDA...")
    print("Installing PyTorch (CUDA 12.1)...")
    run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Install PyTorch"
    )

    # 4. 安装 GPU 库
    print("\n[4/6] Installing GPU libraries...")
    run_command(
        "pip install sentence-transformers FlagEmbedding faiss-gpu scikit-learn",
        "Install GPU libraries"
    )

    # 5. 安装项目依赖
    print("\n[5/6] Installing project dependencies...")
    run_command(
        "pip install -r requirements.txt",
        "Install requirements"
    )

    # 6. 验证 GPU
    print("\n[6/6] Verifying GPU setup...")
    run_command(
        'python -c "import torch; print(\'PyTorch:\', torch.__version__); print(\'CUDA available:\', torch.cuda.is_available())"',
        "Verify GPU"
    )

    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Activate: conda activate research_assistant")
    print("2. Run: python main.py --cluster")
    print()

if __name__ == "__main__":
    main()
