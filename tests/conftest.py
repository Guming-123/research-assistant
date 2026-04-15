"""
Pytest configuration and fixtures
"""

import pytest
import asyncio
import sys
from pathlib import Path
import tempfile
import shutil
import os

# 设置测试用的API key
os.environ["OPENAI_API_KEY"] = "test_key_for_testing"
os.environ["LITELLM_API_KEY"] = "test_key_for_testing"

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def event_loop():
    """创建事件循环"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_workspace():
    """创建临时工作区"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
async def workspace(temp_workspace):
    """创建SharedWorkspace实例"""
    from src.core.workspace import SharedWorkspace

    ws = SharedWorkspace(temp_workspace)
    await ws.load_all()
    yield ws
    # 清理
    shutil.rmtree(temp_workspace, ignore_errors=True)


@pytest.fixture
def sample_paper():
    """示例论文数据"""
    return {
        "paperId": "test123",
        "title": "Test Paper: Deep Learning",
        "authors": [{"name": "Author One"}, {"name": "Author Two"}],
        "abstract": "This is a test abstract about deep learning.",
        "year": 2024,
        "source": "test",
        "url": "https://example.com/paper123",
        "venue": "Test Conference",
        "citationCount": 42,
    }


@pytest.fixture
def sample_papers():
    """示例论文列表"""
    return [
        {
            "paperId": f"paper{i}",
            "title": f"Test Paper {i}: Machine Learning",
            "authors": [{"name": f"Author {i}"}],
            "abstract": f"Abstract for paper {i} about machine learning.",
            "year": 2020 + (i % 5),
            "source": "test",
            "url": f"https://example.com/paper{i}",
            "venue": f"Conference {i % 3}",
            "citationCount": 10 + i,
        }
        for i in range(1, 11)
    ]
