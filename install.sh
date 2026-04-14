#!/bin/bash
# Installation script for Multi-Agent Literature Review System
# 在 WSL/Linux 系统上安装依赖

echo "=== Installing Multi-Agent Literature Review System ==="

# 创建虚拟环境
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# 激活虚拟环境
echo "Activating virtual environment..."
source venv/bin/activate

# 升级 pip
echo "Upgrading pip..."
pip install --upgrade pip

# 安装依赖
echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "Then run the system:"
echo "  python main.py --topic 'your research topic' --full"
echo ""