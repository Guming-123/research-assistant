#!/usr/bin/env python3
"""
Multi-Agent Literature Review System - Main Entry Point
基于论文《AI-Augmented Literature Reviews: Efficient Clustering and Summarization for Researchers》

Usage:
    python main.py --topic "deep learning in computer vision" --full
    python main.py --status
    python main.py --list-papers
"""

import asyncio
import sys
from pathlib import Path

# 首先加载 .env 文件（必须在其他导入之前）
try:
    from dotenv import load_dotenv
    # 查找 .env 文件（从当前目录向上查找）
    env_path = Path(__file__).parent / ".env"
    load_dotenv(env_path, override=True)
except ImportError:
    pass

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.cli import main


if __name__ == "__main__":
    main()
