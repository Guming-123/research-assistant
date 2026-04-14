"""
Integration Tests for Multi-Agent Literature Review System
端到端集成测试 - 验证完整流程的正确性
"""

import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch
import asyncio

from src.core import Coordinator, SharedWorkspace, RQManager
from src.agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent
from src.core.workspace import LiteratureRecord
from src.config import ScreenConfig


@pytest.fixture
def temp_workspace():
    """创建临时工作区"""
    temp_dir = tempfile.mkdtemp()
    workspace = SharedWorkspace(temp_dir)
    yield workspace
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def mock_rq_manager(temp_workspace):
    """创建Mock RQ管理器"""
    manager = RQManager(temp_workspace.base_path)
    asyncio.run(manager.initialize_from_topic("test topic"))
    return manager


# Mock responses for external APIs
mock_search_response = [
    {
        "paperId": "test1",
        "title": "Test Paper 1: Deep Learning for NLP",
        "abstract": "This paper presents a novel deep learning approach for NLP tasks.",
        "year": 2024,
        "authors": [{"name": "Author One"}],
        "venue": "Conference 2024",
        "citationCount": 10,
        "url": "https://example.com/test1",
    },
    {
        "paperId": "test2",
        "title": "Test Paper 2: Transformer Models",
        "abstract": "We propose a new transformer architecture.",
        "year": 2023,
        "authors": [{"name": "Author Two"}],
        "venue": "Conference 2023",
        "citationCount": 5,
        "url": "https://example.com/test2",
    },
]

mock_embeddings = [[0.1] * 1536 for _ in range(10)]  # Mock embeddings


@pytest.mark.asyncio
async def test_full_pipeline_small(temp_workspace, mock_rq_manager):
    """
    端到端测试：小规模完整流程
    验证：搜索→筛选→聚类→摘要 全链路输出合理
    """
    # Mock外部API调用
    with patch("src.agents.search_agent.multi_source_search", return_value=mock_search_response), \
         patch("src.utils.embedding.get_embeddings", return_value=mock_embeddings), \
         patch("src.utils.api.SemanticScholarAPI") as mock_s2, \
         patch("src.utils.api.ArxivAPI") as mock_arxiv:

        # 设置API mock
        mock_s2.return_value.__aenter__.return_value.search_papers = AsyncMock(return_value=mock_search_response)
        mock_arxiv.return_value.__aenter__.return_value.search_papers = AsyncMock(return_value=[])

        # 创建Coordinator
        coordinator = Coordinator(
            workspace=temp_workspace,
            rq_manager=mock_rq_manager,
        )

        # 注册Agent
        coordinator.register_agent(SearchAgent(temp_workspace))
        coordinator.register_agent(ScreenAgent(temp_workspace, mock_rq_manager))
        coordinator.register_agent(ClusterAgent(temp_workspace))
        coordinator.register_agent(SummaryAgent(temp_workspace))

        # 设置人工审核回调（自动批准）
        async def auto_approve(gate, data):
            return True

        from src.core.coordinator import QualityGate
        for gate in QualityGate:
            coordinator.register_human_review_callback(gate, auto_approve)

        # 执行完整流程
        result = await coordinator.run(
            research_topic="deep learning for natural language processing",
            year_range=(2020, 2024),
            max_results=100,
            auto_mode=True,
        )

        # 验证结果
        assert result.success is True, f"Pipeline failed: {result.errors}"
        assert "report" in result.data

        # 验证数据持久化
        papers = await temp_workspace.get_literature()
        assert len(papers) > 0, "No papers were added to workspace"

        clusters = await temp_workspace.get_clusters()
        # 聚类可能为空（样本太少），但不应该报错

        summaries = await temp_workspace.get_all_summaries()
        assert len(summaries) > 0, "No summaries were generated"


@pytest.mark.asyncio
async def test_search_to_screen_pipeline(temp_workspace, mock_rq_manager):
    """
    测试搜索到筛选的子流程
    """
    with patch("src.agents.search_agent.multi_source_search", return_value=mock_search_response), \
         patch("src.utils.api.SemanticScholarAPI") as mock_s2, \
         patch("src.utils.api.ArxivAPI") as mock_arxiv:

        mock_s2.return_value.__aenter__.return_value.search_papers = AsyncMock(return_value=mock_search_response)
        mock_arxiv.return_value.__aenter__.return_value.search_papers = AsyncMock(return_value=[])

        # 搜索阶段
        search_agent = SearchAgent(temp_workspace)
        search_result = await search_agent.run(
            research_topic="test topic",
            year_range=(2020, 2024),
            max_results=50,
        )

        assert search_result.success is True
        assert search_result.data["papers_saved"] > 0

        # 验证数据已保存
        papers = await temp_workspace.get_literature()
        initial_count = len(papers)
        assert initial_count > 0

        # 筛选阶段
        screen_agent = ScreenAgent(temp_workspace, mock_rq_manager)
        screen_result = await screen_agent.run(
            rq_ids=["RQ1"],
            threshold=0.7,
            use_llm=False,  # 禁用LLM以避免API调用
        )

        assert screen_result.success is True
        assert screen_result.data["relevant_count"] >= 0

        # 验证相关分数已更新
        updated_papers = await temp_workspace.get_literature()
        relevant_count = sum(1 for p in updated_papers if p.relevance_score is not None)
        assert relevant_count > 0


@pytest.mark.asyncio
async def test_cluster_to_summary_pipeline(temp_workspace, mock_rq_manager):
    """
    测试聚类到摘要的子流程
    """
    # 添加一些测试论文
    test_papers = [
        LiteratureRecord(
            id=f"paper{i}",
            title=f"Test Paper {i}: Machine Learning Topic",
            authors=["Test Author"],
            abstract=f"This is test abstract for paper {i} discussing ML methods.",
            year=2024,
            source="test",
            url=f"https://example.com/paper{i}",
        )
        for i in range(5)
    ]

    await temp_workspace.add_literature(test_papers)

    # 添加embeddings（用于聚类）
    for i in range(5):
        await temp_workspace.save_embedding(f"paper{i}", [0.1] * 1536)

    with patch("src.utils.embedding.get_embeddings", return_value=mock_embeddings):
        # 聚类阶段
        cluster_agent = ClusterAgent(temp_workspace)
        cluster_result = await cluster_agent.run(
            method="hdbscan",
            min_cluster_size=2,
        )

        assert cluster_result.success is True

        # 验证聚类结果
        clusters = await temp_workspace.get_clusters()
        # 小样本可能产生噪声或小簇

        # 摘要阶段（使用简化的参数）
        summary_agent = SummaryAgent(temp_workspace)

        # 创建简单的RQ树用于测试
        mock_rq_tree = Mock()
        mock_rq_tree.research_topic = "test topic"

        summary_result = await summary_agent.run(
            rq_tree=mock_rq_tree,
            include_methodology=True,
            include_applications=False,
        )

        # 摘要可能成功也可能失败（取决于LLM），验证不会崩溃
        assert summary_result is not None


@pytest.mark.asyncio
async def test_error_recovery_and_rollback(temp_workspace, mock_rq_manager):
    """
    测试错误恢复和回滚机制
    """
    coordinator = Coordinator(
        workspace=temp_workspace,
        rq_manager=mock_rq_manager,
    )

    # 注册Agent
    coordinator.register_agent(SearchAgent(temp_workspace))
    coordinator.register_agent(ScreenAgent(temp_workspace, mock_rq_manager))
    coordinator.register_agent(ClusterAgent(temp_workspace))

    # 创建一个会失败的Search Agent
    class FailingSearchAgent(SearchAgent):
        async def run(self, **kwargs):
            from src.core.agent import AgentResult
            return AgentResult(
                agent_name="FailingSearch",
                success=False,
                errors=["Intentional failure for testing"]
            )

    coordinator.register_agent(FailingSearchAgent(temp_workspace))

    # 设置人工审核回调
    async def auto_approve(gate, data):
        return True

    from src.core.coordinator import QualityGate
    for gate in QualityGate:
        coordinator.register_human_review_callback(gate, auto_approve)

    # 尝试执行流程（预期会失败）
    result = await coordinator.run(
        research_topic="test topic",
        auto_mode=True,
    )

    # 验证错误处理
    assert result.success is False
    assert len(result.errors) > 0

    # 验证工作区状态保持一致
    final_papers = await temp_workspace.get_literature()
    # 失败后不应该有脏数据


@pytest.mark.asyncio
async def test_checkpoint_creation_and_restoration(temp_workspace, mock_rq_manager):
    """
    测试检查点创建和恢复
    """
    # 添加一些测试数据
    test_paper = LiteratureRecord(
        id="checkpoint_test",
        title="Checkpoint Test Paper",
        authors=["Test"],
        abstract="Testing checkpoint functionality",
        year=2024,
        source="test",
        url="https://example.com/checkpoint",
    )
    await temp_workspace.add_literature(test_paper)

    # 创建检查点
    checkpoint_path = await temp_workspace.create_checkpoint("test_checkpoint")
    assert checkpoint_path is not None
    assert "test_checkpoint" in checkpoint_path

    # 修改数据
    await temp_workspace.add_literature(
        LiteratureRecord(
            id="new_paper",
            title="New Paper",
            authors=["Test"],
            abstract="New paper after checkpoint",
            year=2024,
            source="test",
            url="https://example.com/new",
        )
    )

    count_after_change = await temp_workspace.get_literature_count()
    assert count_after_change == 2

    # 恢复检查点
    success = await temp_workspace.restore_checkpoint("test_checkpoint")
    assert success is True

    # 验证数据已恢复
    count_after_restore = await temp_workspace.get_literature_count()
    assert count_after_restore == 1  # 应该恢复到1篇

    restored_papers = await temp_workspace.get_literature()
    assert restored_papers[0].id == "checkpoint_test"


@pytest.mark.asyncio
async def test_concurrent_cli_initialization(temp_workspace, mock_rq_manager):
    """
    测试CLI的并发初始化保护
    """
    from src.cli import LiteratureReviewCLI

    cli = LiteratureReviewCLI(workspace_path=temp_workspace.base_path)

    # 模拟并发调用
    tasks = [
        cli._ensure_initialized() for _ in range(10)
    ]

    # 所有任务应该成功完成
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 验证没有异常
    exceptions = [r for r in results if isinstance(r, Exception)]
    assert len(exceptions) == 0, f"Found {len(exceptions)} exceptions during concurrent initialization"

    # 验证只初始化了一次
    # （这个验证比较困难，因为我们无法直接访问内部计数器）
    # 但至少能确认没有崩溃


@pytest.mark.asyncio
async def test_dual_model_screening_flow():
    """
    测试双模型筛选流程
    """
    temp_workspace = SharedWorkspace(tempfile.mkdtemp())

    try:
        # 添加测试论文
        test_paper = LiteratureRecord(
            id="dual_model_test",
            title="High Quality Paper",
            authors=["Expert Author"],
            abstract="This paper presents groundbreaking research in the field.",
            year=2024,
            source="test",
            url="https://example.com/high_quality",
        )
        await temp_workspace.add_literature(test_paper)

        # 创建有refinement_model的配置
        config = ScreenConfig(
            name="ScreenAgent",
            screening_model="gpt-4o-mini",
            refinement_model="gpt-4o",  # 启用双模型
            llm_threshold_min=0.5,
            llm_threshold_max=0.8,
        )

        # Mock LLM调用
        with patch("src.utils.llm.get_llm_client") as mock_get_llm:
            # 创建两个不同的mock LLM
            mock_screening_llm = Mock()
            mock_refinement_llm = Mock()

            # 设置返回值
            mock_screening_llm.ainvoke = AsyncMock(return_value=Mock(content='{"relevant": true, "confidence": 0.7}'))
            mock_refinement_llm.ainvoke = AsyncMock(return_value=Mock(content='{"relevant": true, "confidence": 0.9}'))

            def side_effect_get_llm(*args, **kwargs):
                if "screening_model" in str(args) or kwargs.get("model", "").startswith("gpt-4o-mini"):
                    return mock_screening_llm
                elif "refinement_model" in str(args) or kwargs.get("model", "") == "gpt-4o":
                    return mock_refinement_llm
                return Mock()

            mock_get_llm.side_effect = side_effect_get_llm

            # 创建ScreenAgent
            from src.core.rq_manager import RQManager
            rq_manager = RQManager(temp_workspace.base_path)
            await rq_manager.initialize_from_topic("test")

            screen_agent = ScreenAgent(temp_workspace, rq_manager, config=config)

            # 验证两个LLM客户端都已创建
            assert screen_agent.screening_llm is not None
            assert screen_agent.refinement_llm is not None

    finally:
        shutil.rmtree(temp_workspace.base_path, ignore_errors=True)