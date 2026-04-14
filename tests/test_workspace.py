"""
Tests for SharedWorkspace
测试工作区的读写、序列化功能
"""

import asyncio
import pytest
import tempfile
import shutil
from pathlib import Path

from src.core.workspace import (
    SharedWorkspace,
    LiteratureRecord,
    ClusterResult,
    WorkspaceEntry,
)


@pytest.fixture
def temp_workspace():
    """创建临时工作区"""
    temp_dir = tempfile.mkdtemp()
    workspace = SharedWorkspace(temp_dir)
    yield workspace
    # 清理
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_workspace_initialization(temp_workspace):
    """测试工作区初始化"""
    assert temp_workspace.base_path.exists()
    assert temp_workspace.get_literature_count() == 0


@pytest.mark.asyncio
async def test_add_literature(temp_workspace):
    """测试添加文献记录"""
    record = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One", "Author Two"],
        abstract="This is a test abstract.",
        year=2024,
        source="semantic_scholar",
        url="https://example.com/paper1",
    )

    count = await temp_workspace.add_literature(record)
    assert count == 1
    assert temp_workspace.get_literature_count() == 1


@pytest.mark.asyncio
async def test_get_literature(temp_workspace):
    """测试获取文献记录"""
    record = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )

    await temp_workspace.add_literature(record)

    # 获取所有文献
    papers = await temp_workspace.get_literature()
    assert len(papers) == 1
    assert papers[0].title == "Test Paper"

    # 通过ID获取
    papers_by_id = await temp_workspace.get_literature(paper_ids=["test1"])
    assert len(papers_by_id) == 1


@pytest.mark.asyncio
async def test_update_literature(temp_workspace):
    """测试更新文献记录"""
    record = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )

    await temp_workspace.add_literature(record)

    # 更新relevance_score
    success = await temp_workspace.update_literature("test1", {"relevance_score": 0.85})
    assert success is True

    papers = await temp_workspace.get_literature(paper_ids=["test1"])
    assert papers[0].relevance_score == 0.85


@pytest.mark.asyncio
async def test_literature_deduplication(temp_workspace):
    """测试文献去重"""
    record1 = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )

    record2 = LiteratureRecord(
        id="test2",  # 不同ID
        title="Test Paper",  # 相同标题
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )

    # 相同标题的记录应该被去重（基于hash）
    count = await temp_workspace.add_literature([record1, record2])
    # 由于hash相同，只应该添加一个
    assert count == 1


@pytest.mark.asyncio
async def test_save_and_load_clusters(temp_workspace):
    """测试聚类结果的保存和加载"""
    cluster = ClusterResult(
        cluster_id=1,
        label="Test Cluster",
        description="A test cluster",
        paper_ids=["paper1", "paper2"],
        representative_papers=[],
        sub_themes=["theme1", "theme2"],
        size=2,
    )

    await temp_workspace.save_clusters([cluster])

    # 获取所有聚类
    clusters = await temp_workspace.get_clusters()
    assert len(clusters) == 1
    assert clusters[0].label == "Test Cluster"

    # 获取单个聚类
    single_cluster = await temp_workspace.get_cluster(1)
    assert single_cluster is not None
    assert single_cluster.label == "Test Cluster"


@pytest.mark.asyncio
async def test_summary_management(temp_workspace):
    """测试摘要管理"""
    await temp_workspace.save_summary("test_key", "Test summary content")

    summary = await temp_workspace.get_summary("test_key")
    assert summary == "Test summary content"

    summaries = await temp_workspace.get_all_summaries()
    assert "test_key" in summaries
    assert summaries["test_key"] == "Test summary content"


@pytest.mark.asyncio
async def test_workspace_persistence(temp_workspace):
    """测试工作区持久化"""
    # 添加数据
    record = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )
    await temp_workspace.add_literature(record)

    # 创建新工作区实例（模拟重启）
    new_workspace = SharedWorkspace(temp_workspace.base_path)
    await new_workspace.load_all()

    # 验证数据已恢复
    assert new_workspace.get_literature_count() == 1
    papers = await new_workspace.get_literature()
    assert papers[0].title == "Test Paper"


@pytest.mark.asyncio
async def test_checkpoint_and_restore(temp_workspace):
    """测试检查点创建和恢复"""
    # 添加一些数据
    record = LiteratureRecord(
        id="test1",
        title="Test Paper",
        authors=["Author One"],
        abstract="Abstract",
        year=2024,
        source="test",
        url="https://example.com",
    )
    await temp_workspace.add_literature(record)

    # 创建检查点
    checkpoint_path = await temp_workspace.create_checkpoint("test_checkpoint")
    assert checkpoint_path is not None
    assert "test_checkpoint" in checkpoint_path

    # 修改数据
    await temp_workspace.add_literature(
        LiteratureRecord(
            id="test2",
            title="Another Paper",
            authors=["Author Two"],
            abstract="Another abstract",
            year=2024,
            source="test",
            url="https://example.com/2",
        )
    )

    # 恢复检查点
    success = await temp_workspace.restore_checkpoint("test_checkpoint")
    assert success is True
    assert temp_workspace.get_literature_count() == 1  # 应该恢复到1篇


def test_workspace_info(temp_workspace):
    """测试获取工作区信息"""
    info = temp_workspace.get_workspace_info()
    assert "literature_count" in info
    assert "cluster_count" in info
    assert "summary_count" in info
    assert info["literature_count"] == 0