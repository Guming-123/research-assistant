@echo off
echo ========================================
echo GPU 环境配置脚本
echo ========================================
echo.

REM 检查 CUDA 版本
echo [1/4] 检查 NVIDIA GPU 和 CUDA...
nvidia-smi
echo.

REM 安装 PyTorch (CUDA 12.1)
echo [2/4] 安装 PyTorch with CUDA 支持...
echo 如果您的 CUDA 版本不是 12.1，请访问 https://pytorch.org/get-started/locally/
echo.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo.

REM 安装其他 GPU 依赖
echo [3/4] 安装 GPU 加速库...
pip install sentence-transformers FlagEmbedding faiss-gpu
echo.

REM 验证 GPU 可用性
echo [4/4] 验证 GPU 配置...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo.

echo ========================================
echo GPU 配置完成！
echo ========================================
echo.
echo 下一步：
echo 1. 重启您的程序
echo 2. 程序会自动使用 GPU 加速
echo 3. 您应该看到日志显示 "Using device: cuda"
echo.
pause
