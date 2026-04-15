"""
Integration Tests
"""

import pytest
import asyncio
from unittest.mock import patch, Mock

from src.core.workspace import SharedWorkspace, LiteratureRecord, ClusterResult
from src.core.rq_manager import RQManager
from src.agents.search_agent import SearchAgent
from src.agents.cluster_agent import ClusterAgent
from src.config import ClusterConfig


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_default_search_queries(self, temp_workspace):
        """测试默认搜索查询"""
        workspace = SharedWorkspace(temp_workspace)
        await workspace.load_all()

        # Mock LLM client to avoid API key requirement
        with patch('src.core.agent.BaseAgent.__init__', return_value=None):
            agent = SearchAgent(workspace)
            agent.workspace = workspace

            # 测试默认查询
            queries = agent._default_search_queries("deep learning")
            assert len(queries) > 0
            assert queries[0]["database"] == "multi"

    @pytest.mark.asyncio
    async def test_clustering_workflow(self, temp_workspace):
        """测试聚类工作流"""
        workspace = SharedWorkspace(temp_workspace)
        await workspace.load_all()

        # 添加测试论文（使用更多样本来触发 PCA 路径）
        import random
        papers = [
            LiteratureRecord(
                id=f"paper{i}",
                title=f"Test Paper {i}",
                authors=["Test Author"],
                abstract=f"Abstract {i} about machine learning.",
                year=2024,
                source="test",
                url=f"https://test.com/paper{i}",
            )
            for i in range(1100)
        ]
        await workspace.add_literature(papers)

        # 添加embeddings（使用随机数据）
        embeddings = [[random.random() for _ in range(300)] for _ in range(1100)]
        for i, emb in enumerate(embeddings):
            await workspace.save_embedding(f"paper{i}", emb)

        # 测试聚类
        # Mock LLM client to avoid API key requirement
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ClusterConfig(
                name="ClusterAgent",
                description="Cluster agent for testing"
            )
            agent = ClusterAgent(workspace, config=config)
            agent.log_progress = Mock()

            # 获取论文
            papers_list = await workspace.get_literature()
            assert len(papers_list) == 1100

            # 降维
            embeddings_list = [embeddings[i] for i in range(1100)]
            reduced = await agent._dimensionality_reduction(embeddings_list)
            assert reduced.shape == (1100, 2)

    @pytest.mark.asyncio
    async def test_rq_tree_workflow(self, temp_workspace):
        """测试RQ树工作流"""
        # 初始化RQ管理器
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("neural networks")

        # 获取根问题
        root_questions = rq_manager.current_tree.root_questions
        assert len(root_questions) > 0

        # 获取一级问题
        level1_questions = rq_manager.current_tree.get_level1_questions()
        assert len(level1_questions) > 0

        # 测试导出
        export = rq_manager.export_for_report()
        assert "research_topic" in export
        assert export["research_topic"] == "neural networks"
        assert "level1" in export

    @pytest.mark.asyncio
    async def test_workspace_persistence(self, temp_workspace):
        """测试工作区持久化"""
        # 创建并添加数据
        workspace1 = SharedWorkspace(temp_workspace)
        await workspace1.load_all()

        paper = LiteratureRecord(
            id="persist_test",
            title="Persistence Test",
            authors=["Test Author"],
            abstract="Testing persistence",
            year=2024,
            source="test",
            url="https://test.com",
        )
        await workspace1.add_literature(paper)

        # 保存摘要
        await workspace1.save_summary("test", "test summary")

        # 创建新实例并加载
        workspace2 = SharedWorkspace(temp_workspace)
        await workspace2.load_all()

        # 验证数据
        assert workspace2.get_literature_count() == 1
        papers = await workspace2.get_literature()
        assert papers[0].id == "persist_test"

        summary = await workspace2.get_summary("test")
        assert summary == "test summary"

    @pytest.mark.asyncio
    async def test_checkpoint_workflow(self, temp_workspace):
        """测试检查点工作流"""
        workspace = SharedWorkspace(temp_workspace)
        await workspace.load_all()

        # 添加数据
        papers = [
            LiteratureRecord(
                id=f"checkpoint_paper{i}",
                title=f"Checkpoint Paper {i}",
                authors=["Test"],
                abstract=f"Abstract {i}",
                year=2024,
                source="test",
                url=f"https://test.com/{i}",
            )
            for i in range(5)
        ]
        await workspace.add_literature(papers)

        # 创建检查点
        checkpoint_path = await workspace.create_checkpoint("before_changes")
        assert checkpoint_path is not None

        # 修改数据
        await workspace.remove_literature(["checkpoint_paper1", "checkpoint_paper2"])
        assert workspace.get_literature_count() == 3

        # 恢复检查点
        success = await workspace.restore_checkpoint("before_changes")
        assert success is True
        assert workspace.get_literature_count() == 5
