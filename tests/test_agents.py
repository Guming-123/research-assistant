"""
Tests for Agents
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.agents.search_agent import SearchAgent
from src.agents.screen_agent import ScreenAgent
from src.agents.cluster_agent import ClusterAgent
from src.agents.summary_agent import SummaryAgent
from src.core.workspace import LiteratureRecord
from src.core.agent import AgentConfig
from src.config import SearchConfig, ScreenConfig, ClusterConfig, SummaryConfig


class TestSearchAgent:
    """测试SearchAgent"""

    def test_agent_initialization(self, workspace):
        """测试初始化"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = SearchConfig(
                name="SearchAgent",
                description="Search agent for testing"
            )
            agent = SearchAgent(workspace, config=config)

            assert agent.name == "SearchAgent"
            assert agent.workspace == workspace

    def test_validate_input(self, workspace):
        """测试输入验证"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = SearchConfig(
                name="SearchAgent",
                description="Search agent for testing"
            )
            agent = SearchAgent(workspace, config=config)

            # 有效输入
            assert agent.validate_input(research_topic="deep learning") is True

            # 无效输入
            assert agent.validate_input() is False

    def test_default_search_queries(self, workspace):
        """测试默认搜索查询"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = SearchConfig(
                name="SearchAgent",
                description="Search agent for testing"
            )
            agent = SearchAgent(workspace, config=config)

            queries = agent._default_search_queries("neural networks")

            assert len(queries) > 0
            assert queries[0]["database"] == "multi"


class TestScreenAgent:
    """测试ScreenAgent"""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, workspace, temp_workspace):
        """测试初始化"""
        from src.core.rq_manager import RQManager

        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("test topic")

        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ScreenConfig(
                name="ScreenAgent",
                description="Screen agent for testing"
            )
            agent = ScreenAgent(workspace, rq_manager, config=config)

            assert agent.name == "ScreenAgent"
            assert agent.workspace == workspace

    def test_validate_input(self, workspace, temp_workspace):
        """测试输入验证"""
        from src.core.rq_manager import RQManager

        rq_manager = RQManager(temp_workspace)

        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ScreenConfig(
                name="ScreenAgent",
                description="Screen agent for testing"
            )
            agent = ScreenAgent(workspace, rq_manager, config=config)

            # 有效输入（需要 rq_ids）
            assert agent.validate_input(rq_ids=["rq1"]) is True

            # 无效输入（缺少 rq_ids）
            assert agent.validate_input() is False


class TestClusterAgent:
    """测试ClusterAgent"""

    def test_agent_initialization(self, workspace):
        """测试初始化"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ClusterConfig(
                name="ClusterAgent",
                description="Cluster agent for testing"
            )
            agent = ClusterAgent(workspace, config=config)

            assert agent.name == "ClusterAgent"
            assert agent.workspace == workspace

    def test_validate_input(self, workspace):
        """测试输入验证"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ClusterConfig(
                name="ClusterAgent",
                description="Cluster agent for testing"
            )
            agent = ClusterAgent(workspace, config=config)

            # ClusterAgent 的 validate_input 总是返回 True
            assert agent.validate_input() is True

    @pytest.mark.asyncio
    async def test_dimensionality_reduction(self, workspace):
        """测试降维"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = ClusterConfig(
                name="ClusterAgent",
                description="Cluster agent for testing"
            )
            agent = ClusterAgent(workspace, config=config)
            agent.log_progress = Mock()

            # 使用更多样本（>1000）来触发 PCA 路径，避免 t-SNE 栈溢出
            import random
            embeddings = [[random.random() for _ in range(300)] for _ in range(1100)]

            reduced = await agent._dimensionality_reduction(embeddings)

            assert reduced is not None
            assert reduced.shape[0] == 1100  # 1100个样本
            assert reduced.shape[1] == 2  # 降维到2维


class TestSummaryAgent:
    """测试SummaryAgent"""

    def test_agent_initialization(self, workspace):
        """测试初始化"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = SummaryConfig(
                name="SummaryAgent",
                description="Summary agent for testing"
            )
            agent = SummaryAgent(workspace, config=config)

            assert agent.name == "SummaryAgent"
            assert agent.workspace == workspace

    def test_validate_input(self, workspace):
        """测试输入验证"""
        # 使用 mock 避免 API key 要求
        with patch('src.utils.llm.get_llm_client') as mock_llm:
            mock_llm.return_value = Mock()

            config = SummaryConfig(
                name="SummaryAgent",
                description="Summary agent for testing"
            )
            agent = SummaryAgent(workspace, config=config)

            # 有效输入（需要 rq_tree）
            assert agent.validate_input(rq_tree=Mock()) is True

            # 无效输入（缺少 rq_tree）
            assert agent.validate_input() is False
