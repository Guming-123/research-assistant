"""
Tests for SearchAgent
测试搜索Agent的API调用和数据处理
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import tempfile
import shutil

from src.agents.search_agent import SearchAgent
from src.core.workspace import SharedWorkspace
from src.config import SearchConfig


@pytest.fixture
def temp_workspace():
    """创建临时工作区"""
    temp_dir = tempfile.mkdtemp()
    workspace = SharedWorkspace(temp_dir)
    yield workspace
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_search_agent_initialization(temp_workspace):
    """测试SearchAgent初始化"""
    config = SearchConfig(name="TestSearch")
    agent = SearchAgent(temp_workspace, config=config)

    assert agent.name == "SearchAgent"
    assert agent.config is not None


@pytest.mark.asyncio
async def test_query_building(temp_workspace):
    """测试查询构建逻辑"""
    agent = SearchAgent(temp_workspace)

    queries = await agent._build_search_queries(
        research_topic="deep learning in computer vision",
        scope_description="AI applications in image processing",
    )

    assert len(queries) > 0
    assert all("database" in q for q in queries)
    assert all("query" in q for q in queries)


@pytest.mark.asyncio
async def test_default_search_queries(temp_workspace):
    """测试默认搜索查询"""
    agent = SearchAgent(temp_workspace)

    queries = agent._default_search_queries("machine learning")

    assert len(queries) > 0
    assert queries[0]["database"] == "semantic_scholar"


@pytest.mark.asyncio
async def test_paper_deduplication(temp_workspace):
    """测试论文去重逻辑"""
    # 模拟多源检索的重复数据
    papers = [
        {
            "paperId": "source1_123",
            "title": "Same Paper",
            "abstract": "Same content",
            "year": 2024,
            "authors": [{"name": "Author"}],
            "url": "https://example.com/1",
        },
        {
            "paperId": "source2_456",
            "title": "Same Paper",  # 相同标题
            "abstract": "Same content",
            "year": 2024,
            "authors": [{"name": "Author"}],
            "url": "https://example.com/2",
        },
        {
            "paperId": "unique_789",
            "title": "Different Paper",
            "abstract": "Different content",
            "year": 2023,
            "authors": [{"name": "Other"}],
            "url": "https://example.com/3",
        },
    ]

    from src.utils.api import deduplicate_papers

    unique_papers = deduplicate_papers(papers)

    # 应该去重为2篇（相同标题的2篇 + 1篇不同的）
    assert len(unique_papers) == 2


@pytest.mark.asyncio
async def test_standardize_records(temp_workspace):
    """测试记录标准化"""
    agent = SearchAgent(temp_workspace)

    raw_paper = {
        "paperId": "test123",
        "title": "  Test Paper with Spaces  ",
        "abstract": "Test abstract",
        "year": 2024,
        "authors": [{"name": "First Author"}, {"name": "Second Author"}],
        "venue": "CVPR 2024",
        "citationCount": 42,
        "url": "https://example.com/test123",
        "doi": "10.1234/test.5678",
    }

    records = await agent._standardize_records([raw_paper])

    assert len(records) == 1
    record = records[0]

    # 验证字段标准化
    assert record.id == "test123"
    assert record.title.strip() == "Test Paper with Spaces"  # 应该strip
    assert len(record.authors) == 2
    assert record.venue == "CVPR 2024"
    assert record.citation_count == 42
    assert record.doi == "10.1234/test.5678"


@pytest.mark.asyncio
async def test_search_metrics(temp_workspace):
    """测试搜索指标统计"""
    agent = SearchAgent(temp_workspace)

    metrics = await agent.get_search_statistics()

    assert "total" in metrics
    assert "sources" in metrics
    assert "years" in metrics
    assert "venues" in metrics


def test_search_config_defaults():
    """测试SearchConfig默认值"""
    config = SearchConfig(name="TestSearch")

    assert config.default_year_start == 2018
    assert config.default_year_end == 2025
    assert config.default_max_results == 500
    assert config.rate_limit_per_second == 10
    assert config.enable_pdf_download is False