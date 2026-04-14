"""
Tests for RQManager
测试RQ层级构建、查询功能
"""

import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.rq_manager import (
    RQManager,
    RQTree,
    ResearchQuestion,
    RQLevel,
)


@pytest.fixture
def temp_rq_manager():
    """创建临时RQ管理器"""
    temp_dir = tempfile.mkdtemp()
    manager = RQManager(temp_dir)
    yield manager
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_initialize_default_rqs(temp_rq_manager):
    """测试默认RQ的初始化"""
    tree = await temp_rq_manager.initialize_from_topic("deep learning")

    assert tree is not None
    assert tree.research_topic == "deep learning"
    assert len(tree.root_questions) == 3  # 应该有3个一级RQ


@pytest.mark.asyncio
async def test_rq_hierarchy_structure(temp_rq_manager):
    """测试RQ层级结构"""
    tree = await temp_rq_manager.initialize_from_topic("machine learning")

    # 验证一级RQ
    level1_questions = tree.get_level1_questions()
    assert len(level1_questions) == 3
    assert all(rq.level == RQLevel.LEVEL_1 for rq in level1_questions)

    # 验证二级RQ
    level2_questions = tree.get_level2_questions()
    assert len(level2_questions) >= 6  # 至少6个二级RQ
    assert all(rq.level == RQLevel.LEVEL_2 for rq in level2_questions)

    # 验证三级RQ
    level3_questions = tree.get_level3_questions()
    assert len(level3_questions) >= 2  # 至少有2个三级RQ
    assert all(rq.level == RQLevel.LEVEL_3 for rq in level3_questions)


@pytest.mark.asyncio
async def test_get_question_by_id(temp_rq_manager):
    """测试通过ID获取RQ"""
    tree = await temp_rq_manager.initialize_from_topic("computer vision")

    # 获取RQ1
    rq1 = temp_rq_manager.get_question("RQ1")
    assert rq1 is not None
    assert rq1.level == RQLevel.LEVEL_1
    assert rq1.id == "RQ1"

    # 获取不存在的RQ
    rq999 = temp_rq_manager.get_question("RQ999")
    assert rq999 is None


@pytest.mark.asyncio
async def test_rq_children(temp_rq_manager):
    """测试RQ的子节点"""
    tree = await temp_rq_manager.initialize_from_topic("natural language processing")

    rq1 = temp_rq_manager.get_question("RQ1")
    assert rq1 is not None
    assert len(rq1.children) >= 2  # RQ1应该有子节点

    # 验证子节点的父关系
    for child in rq1.children:
        assert child.parent_id == "RQ1"


@pytest.mark.asyncio
async def test_get_all_descendants(temp_rq_manager):
    """测试获取所有后代RQ"""
    tree = await temp_rq_manager.initialize_from_topic("robotics")

    rq1 = temp_rq_manager.get_question("RQ1")
    descendants = rq1.get_all_descendants()

    assert len(descendants) >= 2  # RQ1应该有后代
    # 验证后代不包含自己
    assert all(d.id != "RQ1" for d in descendants)


@pytest.mark.asyncio
async def test_rq_keywords(temp_rq_manager):
    """测试RQ关键词"""
    tree = await temp_rq_manager.initialize_from_topic("data science")

    rq1 = temp_rq_manager.get_question("RQ1")
    assert rq1 is not None
    assert len(rq1.keywords) > 0
    # 关键词应该包含方法相关的词
    assert any(keyword in rq1.keywords for keyword in ["method", "approach", "technique"])


@pytest.mark.asyncio
async def test_rq_status_management(temp_rq_manager):
    """测试RQ状态管理"""
    tree = await temp_rq_manager.initialize_from_topic("artificial intelligence")

    # 初始状态
    rq1 = temp_rq_manager.get_question("RQ1")
    assert rq1.status == "pending"

    # 更新状态
    success = temp_rq_manager.update_question_status("RQ1", "in_progress")
    assert success is True

    # 验证状态已更新
    rq1_updated = temp_rq_manager.get_question("RQ1")
    assert rq1_updated.status == "in_progress"

    # 尝试更新不存在的RQ
    success = temp_rq_manager.update_question_status("RQ999", "completed")
    assert success is False


@pytest.mark.asyncio
async def test_generate_search_queries(temp_rq_manager):
    """测试生成搜索查询"""
    tree = await temp_rq_manager.initialize_from_topic("deep learning")

    queries = temp_rq_manager.generate_search_queries("RQ1")
    assert len(queries) > 0

    # 验证查询格式
    for query in queries:
        assert isinstance(query, str)
        assert len(query) > 0


@pytest.mark.asyncio
async def test_tree_persistence(temp_rq_manager):
    """测试RQ树的持久化"""
    # 创建并保存RQ树
    tree = await temp_rq_manager.initialize_from_topic("test topic")

    # 创建新的管理器实例（模拟重启）
    new_manager = RQManager(temp_rq_manager.workspace_path)
    loaded_tree = await new_manager.load()

    assert loaded_tree is not None
    assert loaded_tree.research_topic == "test topic"
    assert len(loaded_tree.root_questions) == 3


@pytest.mark.asyncio
async def test_custom_rqs(temp_rq_manager):
    """测试自定义RQ"""
    custom_rqs = [
        {
            "id": "CUST1",
            "question": "Custom research question 1",
            "level": 1,
            "keywords": ["custom", "question"],
            "children": [],
        },
        {
            "id": "CUST2",
            "question": "Custom research question 2",
            "level": 1,
            "keywords": ["another", "question"],
            "children": [],
        },
    ]

    tree = await temp_rq_manager.initialize_from_topic(
        "custom topic",
        custom_rqs=custom_rqs,
    )

    assert tree is not None
    assert len(tree.root_questions) == 2
    assert tree.root_questions[0].id == "CUST1"


def test_rq_to_dict():
    """测试RQ的序列化"""
    rq = ResearchQuestion(
        id="TEST1",
        question="Test question",
        level=RQLevel.LEVEL_2,
        keywords=["test", "question"],
    )

    rq_dict = rq.to_dict()

    assert rq_dict["id"] == "TEST1"
    assert rq_dict["question"] == "Test question"
    assert rq_dict["level"] == 2  # 应该是整数
    assert "keywords" in rq_dict


def test_tree_to_dict(temp_rq_manager):
    """测试RQ树的序列化"""
    rq = ResearchQuestion(
        id="ROOT1",
        question="Root question",
        level=RQLevel.LEVEL_1,
        children=[
            ResearchQuestion(
                id="CHILD1",
                question="Child question",
                level=RQLevel.LEVEL_2,
            )
        ],
    )

    tree = RQTree(
        research_topic="test",
        root_questions=[rq],
    )

    tree_dict = tree.to_dict()

    assert tree_dict["research_topic"] == "test"
    assert "root_questions" in tree_dict
    assert len(tree_dict["root_questions"]) == 1


@pytest.mark.asyncio
async def test_export_for_report(temp_rq_manager):
    """测试导出报告格式"""
    await temp_rq_manager.initialize_from_topic("report test")

    export = temp_rq_manager.export_for_report()

    assert "research_topic" in export
    assert "structure" in export
    assert "level1" in export
    assert "level2" in export
    assert "level3" in export