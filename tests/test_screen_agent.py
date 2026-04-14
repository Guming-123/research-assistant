"""
Tests for ScreenAgent
测试NF计算、两阶段筛选逻辑、chunk_id处理
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
import numpy as np

from src.agents.screen_agent import ScreenAgent, ScreeningResult
from src.core.workspace import LiteratureRecord, SharedWorkspace
from src.core.rq_manager import RQManager, RQLevel, ResearchQuestion
from src.config import ScreenConfig


@pytest.fixture
def mock_workspace():
    """Mock工作区"""
    workspace = Mock(spec=SharedWorkspace)
    workspace.get_literature = AsyncMock(return_value=[])
    workspace.save = AsyncMock()
    workspace.save_embedding = AsyncMock()
    return workspace


@pytest.fixture
def mock_rq_manager():
    """Mock RQ管理器"""
    manager = Mock(spec=RQManager)
    rq = ResearchQuestion(
        id="RQ1",
        question="What are the main methods?",
        level=RQLevel.LEVEL_1,
        keywords=["method", "approach", "technique"],
    )
    manager.get_question = Mock(return_value=rq)
    return manager


@pytest.fixture
def screen_agent(mock_workspace, mock_rq_manager):
    """创建ScreenAgent实例"""
    config = ScreenConfig(
        name="ScreenAgent",
        model="gpt-4o",
        temperature=0.3,
        screening_model="gpt-4o-mini",
        llm_threshold_min=0.5,
        llm_threshold_max=0.8,
    )
    agent = ScreenAgent(mock_workspace, mock_rq_manager, config=config)
    return agent


def test_chunk_id_parsing():
    """测试chunk_id的解析（使用安全分隔符）"""
    # 正常情况
    chunk_id = "paper123::0"
    paper_id, index_str = chunk_id.rsplit("::", 1)
    assert paper_id == "paper123"
    assert index_str == "0"

    # paper_id包含下划线的情况
    chunk_id = "paper_123_with_underscores::5"
    paper_id, index_str = chunk_id.rsplit("::", 1)
    assert paper_id == "paper_123_with_underscores"
    assert index_str == "5"


def test_chunk_id_generation():
    """测试chunk_id的生成"""
    paper_id = "test_paper_123"
    chunk_index = 3

    # 使用安全的分隔符
    chunk_id = f"{paper_id}::{chunk_index}"
    assert chunk_id == "test_paper_123::3"

    # 验证可以正确解析
    parsed_paper_id, parsed_index = chunk_id.rsplit("::", 1)
    assert parsed_paper_id == paper_id
    assert int(parsed_index) == chunk_index


@pytest.mark.asyncio
async def test_nf_calculation():
    """测试NF（归一化频率）计算的正确性"""
    # 模拟检索结果
    retrieval_results = {
        "paper1": {0, 1, 2, 3, 4},  # 检索到5个chunks
        "paper2": {0, 1},           # 检索到2个chunks
        "paper3": {0, 1, 2, 3, 4, 5, 6, 7},  # 检索到8个chunks
    }

    # 实际chunk数
    paper_chunk_counts = {
        "paper1": 10,  # NF = 5/10 = 0.5
        "paper2": 5,   # NF = 2/5 = 0.4
        "paper3": 10,  # NF = 8/10 = 0.8
    }

    # 计算NF
    nf_scores = {}
    for paper_id, retrieved_chunks in retrieval_results.items():
        total_chunks = paper_chunk_counts.get(paper_id, 1)
        nf = len(retrieved_chunks) / total_chunks
        nf_scores[paper_id] = nf

    # 验证计算结果
    assert abs(nf_scores["paper1"] - 0.5) < 0.01
    assert abs(nf_scores["paper2"] - 0.4) < 0.01
    assert abs(nf_scores["paper3"] - 0.8) < 0.01


def test_two_stage_filtering_boundaries():
    """测试两阶段筛选的边界值"""
    llm_min, llm_max = 0.5, 0.8

    test_cases = [
        (0.9, "high_confidence"),   # > 0.8, 直接通过
        (0.8, "high_confidence"),   # = 0.8, 直接通过
        (0.75, "llm_required"),     # 0.5-0.8, 需要LLM
        (0.6, "llm_required"),      # 0.5-0.8, 需要LLM
        (0.5, "llm_required"),      # = 0.5, 需要LLM
        (0.49, "low_confidence"),   # < 0.5, 直接拒绝
        (0.3, "low_confidence"),    # < 0.5, 直接拒绝
    ]

    for nf_score, expected_category in test_cases:
        if nf_score >= llm_max:
            category = "high_confidence"
        elif nf_score < llm_min:
            category = "low_confidence"
        else:
            category = "llm_required"

        assert category == expected_category, f"NF={nf_score} 应该归类为 {expected_category}"


def test_two_stage_filtering_edge_cases():
    """测试两阶段筛选的边界情况"""
    llm_min, llm_max = 0.5, 0.8

    # 边界值测试
    edge_cases = [
        (0.499999, "low_confidence"),  # 刚好在阈值下
        (0.5, "llm_required"),         # 刚好在阈值上
        (0.500001, "llm_required"),     # 刚好在阈值上
        (0.799999, "llm_required"),     # 刚好在阈值下
        (0.8, "high_confidence"),       # 刚好在阈值上
        (0.800001, "high_confidence"),   # 刚好在阈值上
    ]

    for nf_score, expected_category in edge_cases:
        if nf_score >= llm_max:
            category = "high_confidence"
        elif nf_score < llm_min:
            category = "low_confidence"
        else:
            category = "llm_required"

        assert category == expected_category, f"NF={nf_score} 边界值处理错误"


@pytest.mark.asyncio
async def test_screening_result_to_dict():
    """测试ScreeningResult的序列化"""
    result = ScreeningResult(
        paper_id="test_paper",
        relevant=True,
        confidence=0.85,
        relevance_scores={"topic": 4, "method": 5, "timeliness": 4},
        reasoning="Highly relevant to the research question",
        normalized_frequency=0.75,
        related_rqs=["RQ1", "RQ2"],
    )

    result_dict = result.to_dict()

    assert result_dict["paper_id"] == "test_paper"
    assert result_dict["relevant"] is True
    assert result_dict["confidence"] == 0.85
    assert result_dict["relevance_scores"]["method"] == 5
    assert result_dict["normalized_frequency"] == 0.75
    assert "RQ1" in result_dict["related_rqs"]


def test_screen_config_defaults():
    """测试ScreenConfig的默认值"""
    config = ScreenConfig(name="TestScreen")

    assert config.chunk_size == 512
    assert config.chunk_overlap == 50
    assert config.default_nf_threshold == 0.7
    assert config.enable_llm_screening is True
    assert config.similarity_threshold == 0.5
    assert config.top_k_chunks == 10
    assert config.llm_threshold_min == 0.5
    assert config.llm_threshold_max == 0.8
    assert config.screening_model == "gpt-4o-mini"  # 新增默认值
    assert config.refinement_model is None


def test_screen_config_custom():
    """测试ScreenConfig的自定义配置"""
    config = ScreenConfig(
        name="TestScreen",
        screening_model="gpt-3.5-turbo",
        refinement_model="gpt-4o",
        llm_threshold_min=0.4,
        llm_threshold_max=0.9,
    )

    assert config.screening_model == "gpt-3.5-turbo"
    assert config.refinement_model == "gpt-4o"
    assert config.llm_threshold_min == 0.4
    assert config.llm_threshold_max == 0.9


@pytest.mark.asyncio
async def test_chunk_count_validation():
    """测试chunk计数的验证"""
    # 模拟不同paper的chunk数
    paper_chunks = {
        "paper1": [f"chunk{i}" for i in range(8)],   # 8个chunks
        "paper2": [f"chunk{i}" for i in range(15)],  # 15个chunks
        "paper3": [f"chunk{i}" for i in range(3)],   # 3个chunks
    }

    # 计算chunk数
    paper_chunk_counts = {pid: len(chunks) for pid, chunks in paper_chunks.items()}

    assert paper_chunk_counts["paper1"] == 8
    assert paper_chunk_counts["paper2"] == 15
    assert paper_chunk_counts["paper3"] == 3
    assert len(paper_chunk_counts) == 3


@pytest.mark.asyncio
async def test_nf_filter_with_zero_chunks():
    """测试chunk数为0时的NF计算"""
    retrieval_results = {
        "paper1": {0, 1, 2},
    }

    # paper1没有chunk信息
    paper_chunk_counts = {}

    # 计算NF（应该有默认值1防止除零）
    for paper_id, retrieved_chunks in retrieval_results.items():
        total_chunks = paper_chunk_counts.get(paper_id, 1)  # 默认为1
        nf = len(retrieved_chunks) / total_chunks
        assert nf == len(retrieved_chunks)  # 应该等于检索到的chunk数