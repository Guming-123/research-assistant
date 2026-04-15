"""
Tests for RQManager
"""

import pytest
import asyncio
from pathlib import Path

from src.core.rq_manager import RQManager, RQLevel, ResearchQuestion


class TestResearchQuestion:
    """测试ResearchQuestion"""

    def test_create_question(self):
        """测试创建研究问题"""
        rq = ResearchQuestion(
            id="rq1",
            level=RQLevel.LEVEL_1,
            question="What is deep learning?",
        )
        assert rq.id == "rq1"
        assert rq.level == RQLevel.LEVEL_1
        assert rq.question == "What is deep learning?"

    def test_to_dict(self):
        """测试转换为字典"""
        rq = ResearchQuestion(
            id="rq1",
            level=RQLevel.LEVEL_1,
            question="What is deep learning?",
        )
        data = rq.to_dict()
        assert data["id"] == "rq1"
        assert data["level"] == 1
        assert data["question"] == "What is deep learning?"


class TestRQManager:
    """测试RQManager"""

    @pytest.mark.asyncio
    async def test_initialization(self, temp_workspace):
        """测试初始化"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("machine learning")

        assert rq_manager.current_tree is not None
        assert rq_manager.current_tree.research_topic == "machine learning"

    @pytest.mark.asyncio
    async def test_get_root_questions(self, temp_workspace):
        """测试获取根问题"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("computer vision")

        root_questions = rq_manager.current_tree.root_questions
        assert len(root_questions) > 0

    @pytest.mark.asyncio
    async def test_get_level_1_questions(self, temp_workspace):
        """测试获取一级问题"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("natural language processing")

        level1_questions = rq_manager.current_tree.get_level1_questions()
        assert len(level1_questions) > 0

    @pytest.mark.asyncio
    async def test_get_level_2_questions(self, temp_workspace):
        """测试获取二级问题"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("deep learning")

        level2_questions = rq_manager.current_tree.get_level2_questions()
        # 可能为空或有一些问题
        assert isinstance(level2_questions, list)

    @pytest.mark.asyncio
    async def test_save_and_load(self, temp_workspace):
        """测试保存和加载"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("reinforcement learning")

        # 保存
        await rq_manager.save()

        # 创建新实例并加载
        new_manager = RQManager(temp_workspace)
        await new_manager.load()

        assert new_manager.current_tree is not None
        assert new_manager.current_tree.research_topic == "reinforcement learning"

    @pytest.mark.asyncio
    async def test_export_for_report(self, temp_workspace):
        """测试导出为报告格式"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("transformer models")

        export_data = rq_manager.export_for_report()
        assert "research_topic" in export_data
        assert "level1" in export_data
        assert len(export_data["level1"]) > 0

    @pytest.mark.asyncio
    async def test_get_question_by_id(self, temp_workspace):
        """测试根据ID获取问题"""
        rq_manager = RQManager(temp_workspace)
        await rq_manager.initialize_from_topic("attention mechanisms")

        # 获取根问题
        questions = rq_manager.current_tree.root_questions
        if questions:
            first_id = questions[0].id
            question = rq_manager.get_question(first_id)
            assert question is not None
            assert question.id == first_id
