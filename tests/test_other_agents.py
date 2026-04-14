"""
Tests for ClusterAgent and SummaryAgent
测试聚类和摘要Agent
"""

import pytest
import tempfile
import shutil
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.agents.cluster_agent import ClusterAgent
from src.agents.summary_agent import SummaryAgent
from src.core.workspace import SharedWorkspace, ClusterResult, LiteratureRecord
from src.config import ClusterConfig, SummaryConfig


@pytest.fixture
def temp_workspace():
    """创建临时工作区"""
    temp_dir = tempfile.mkdtemp()
    workspace = SharedWorkspace(temp_dir)
    yield workspace
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_cluster_agent_initialization(temp_workspace):
    """测试ClusterAgent初始化"""
    config = ClusterConfig(name="TestCluster")
    agent = ClusterAgent(temp_workspace, config=config)

    assert agent.name == "ClusterAgent"


@pytest.mark.asyncio
async def test_nf_filter_with_actual_counts(temp_workspace):
    """
    测试NF计算使用实际chunk计数
    这是v1.2修复的关键功能
    """
    # 模拟检索结果
    retrieval_results = {
        "paper1": {0, 1, 2, 3, 4},  # 5个chunks
        "paper2": {0, 1},           # 2个chunks
    }

    # 实际chunk数
    paper_chunk_counts = {
        "paper1": 10,  # NF = 5/10 = 0.5
        "paper2": 4,   # NF = 2/4 = 0.5
    }

    agent = ClusterAgent(temp_workspace)

    # 应用NF过滤
    filtered = agent._apply_nf_filter(
        retrieval_results,
        paper_chunk_counts,
        threshold=0.4
    )

    # 验证计算结果
    assert "paper1" in filtered
    assert "paper2" in filtered
    assert abs(filtered["paper1"] - 0.5) < 0.01
    assert abs(filtered["paper2"] - 0.5) < 0.01


@pytest.mark.asyncio
async def test_cluster_labeling(temp_workspace):
    """
    测试聚类标签生成逻辑
    """
    agent = ClusterAgent(temp_workspace)

    # 创建测试簇
    cluster = ClusterResult(
        cluster_id=1,
        label="",
        description="",
        paper_ids=["paper1", "paper2"],
        representative_papers=[],
        sub_themes=[],
        size=2,
    )

    # Mock LLM调用
    with patch("src.utils.llm.get_llm_client") as mock_get_llm:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=Mock(content='{"cluster_label": "Test Label", "core_theme": "Test theme"}')
        )
        mock_get_llm.return_value = mock_llm

        labeled_clusters = await agent._label_clusters([cluster])

        assert len(labeled_clusters) == 1
        assert labeled_clusters[0].label == "Test Label"


@pytest.mark.asyncio
async def test_summary_agent_initialization(temp_workspace):
    """测试SummaryAgent初始化"""
    config = SummaryConfig(name="TestSummary")
    agent = SummaryAgent(temp_workspace, config=config)

    assert agent.name == "SummaryAgent"


@pytest.mark.asyncio
async def test_summary_report_structure(temp_workspace):
    """
    测试摘要报告结构
    """
    agent = SummaryAgent(temp_workspace)

    # Mock LLM调用
    with patch("src.utils.llm.get_llm_client") as mock_get_llm:
        mock_llm = Mock()
        mock_llm.ainvoke = AsyncMock(
            return_value=Mock(content="# Test Report\n\nThis is a test summary.")
        )
        mock_get_llm.return_value = mock_llm

        # 创建Mock RQ树
        mock_rq_tree = Mock()
        mock_rq_tree.research_topic = "test topic"
        mock_rq_tree.to_dict = Mock(return_value={"research_topic": "test topic"})

        # 添加测试聚类
        test_cluster = ClusterResult(
            cluster_id=1,
            label="Test Cluster",
            description="Test cluster for summary",
            paper_ids=["paper1"],
            representative_papers=[],
            sub_themes=["theme1"],
            size=1,
        )

        await temp_workspace.save_clusters([test_cluster])

        result = await agent.run(
            rq_tree=mock_rq_tree,
            include_methodology=True,
            include_applications=False,
        )

        assert result.success is True
        assert "report_path" in result.data


def test_cluster_config_defaults():
    """测试ClusterConfig默认值"""
    config = ClusterConfig(name="TestCluster")

    assert config.method == "hdbscan"
    assert config.min_cluster_size == 5
    assert config.dimensionality_reduction == "tsne"
    assert config.n_components == 2


def test_summary_config_defaults():
    """测试SummaryConfig默认值"""
    config = SummaryConfig(name="TestSummary")

    assert config.max_papers_per_cluster == 20
    assert config.include_methodology is True
    assert config.include_applications is True


@pytest.mark.asyncio
async def test_save_report(temp_workspace):
    """
    测试报告保存
    """
    agent = SummaryAgent(temp_workspace)

    test_report = "# Test Report\n\nThis is a test report."

    report_path = await agent._save_report(test_report)

    assert report_path is not None
    assert report_path.endswith(".md")

    # 验证文件已创建
    from pathlib import Path
    report_file = Path(report_path)
    assert report_file.exists()

    # 验证内容
    content = report_file.read_text()
    assert "Test Report" in content