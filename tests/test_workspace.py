"""
Tests for SharedWorkspace
"""

import pytest
import asyncio
from pathlib import Path

from src.core.workspace import SharedWorkspace, LiteratureRecord, ClusterResult


class TestLiteratureRecord:
    """测试LiteratureRecord"""

    def test_create_record(self, sample_paper):
        """测试创建文献记录"""
        record = LiteratureRecord(
            id=sample_paper["paperId"],
            title=sample_paper["title"],
            authors=[a["name"] for a in sample_paper["authors"]],
            abstract=sample_paper["abstract"],
            year=sample_paper["year"],
            source=sample_paper["source"],
            url=sample_paper["url"],
        )
        assert record.id == sample_paper["paperId"]
        assert record.title == sample_paper["title"]
        assert len(record.authors) == 2

    def test_to_dict(self, sample_paper):
        """测试转换为字典"""
        record = LiteratureRecord(
            id=sample_paper["paperId"],
            title=sample_paper["title"],
            authors=[a["name"] for a in sample_paper["authors"]],
            abstract=sample_paper["abstract"],
            year=sample_paper["year"],
            source=sample_paper["source"],
            url=sample_paper["url"],
        )
        data = record.to_dict()
        assert data["id"] == sample_paper["paperId"]
        assert data["title"] == sample_paper["title"]

    def test_from_dict(self, sample_paper):
        """测试从字典创建"""
        data = {
            "id": sample_paper["paperId"],
            "title": sample_paper["title"],
            "authors": [a["name"] for a in sample_paper["authors"]],
            "abstract": sample_paper["abstract"],
            "year": sample_paper["year"],
            "source": sample_paper["source"],
            "url": sample_paper["url"],
        }
        record = LiteratureRecord.from_dict(data)
        assert record.id == sample_paper["paperId"]
        assert record.title == sample_paper["title"]


class TestSharedWorkspace:
    """测试SharedWorkspace"""

    @pytest.mark.asyncio
    async def test_workspace_initialization(self, temp_workspace):
        """测试工作区初始化"""
        ws = SharedWorkspace(temp_workspace)
        assert ws.base_path == Path(temp_workspace)
        assert ws.get_literature_count() == 0

        # 检查目录是否创建
        assert (Path(temp_workspace) / "literature").exists()
        assert (Path(temp_workspace) / "clusters").exists()

    @pytest.mark.asyncio
    async def test_add_literature(self, workspace, sample_paper):
        """测试添加文献"""
        record = LiteratureRecord(
            id=sample_paper["paperId"],
            title=sample_paper["title"],
            authors=[a["name"] for a in sample_paper["authors"]],
            abstract=sample_paper["abstract"],
            year=sample_paper["year"],
            source=sample_paper["source"],
            url=sample_paper["url"],
        )

        count = await workspace.add_literature(record)
        assert count == 1
        assert workspace.get_literature_count() == 1

    @pytest.mark.asyncio
    async def test_add_multiple_literature(self, workspace, sample_papers):
        """测试批量添加文献"""
        records = []
        for paper in sample_papers:
            record = LiteratureRecord(
                id=paper["paperId"],
                title=paper["title"],
                authors=[a["name"] for a in paper["authors"]],
                abstract=paper["abstract"],
                year=paper["year"],
                source=paper["source"],
                url=paper["url"],
            )
            records.append(record)

        count = await workspace.add_literature(records)
        assert count == 10
        assert workspace.get_literature_count() == 10

    @pytest.mark.asyncio
    async def test_get_literature(self, workspace, sample_papers):
        """测试获取文献"""
        # 添加文献
        records = [
            LiteratureRecord(
                id=p["paperId"],
                title=p["title"],
                authors=[a["name"] for a in p["authors"]],
                abstract=p["abstract"],
                year=p["year"],
                source=p["source"],
                url=p["url"],
            )
            for p in sample_papers
        ]
        await workspace.add_literature(records)

        # 获取所有文献
        papers = await workspace.get_literature()
        assert len(papers) == 10

        # 按ID获取
        specific = await workspace.get_literature(paper_ids=["paper1", "paper2"])
        assert len(specific) == 2
        assert specific[0].id == "paper1"

    @pytest.mark.asyncio
    async def test_update_literature(self, workspace, sample_paper):
        """测试更新文献"""
        record = LiteratureRecord(
            id=sample_paper["paperId"],
            title=sample_paper["title"],
            authors=[a["name"] for a in sample_paper["authors"]],
            abstract=sample_paper["abstract"],
            year=sample_paper["year"],
            source=sample_paper["source"],
            url=sample_paper["url"],
        )
        await workspace.add_literature(record)

        # 更新
        success = await workspace.update_literature(
            sample_paper["paperId"], {"citation_count": 100}
        )
        assert success is True

        papers = await workspace.get_literature(paper_ids=[sample_paper["paperId"]])
        assert papers[0].citation_count == 100

    @pytest.mark.asyncio
    async def test_remove_literature(self, workspace, sample_papers):
        """测试删除文献"""
        records = [
            LiteratureRecord(
                id=p["paperId"],
                title=p["title"],
                authors=[a["name"] for a in p["authors"]],
                abstract=p["abstract"],
                year=p["year"],
                source=p["source"],
                url=p["url"],
            )
            for p in sample_papers[:5]
        ]
        await workspace.add_literature(records)

        # 删除
        count = await workspace.remove_literature(["paper1", "paper2"])
        assert count == 2
        assert workspace.get_literature_count() == 3

    @pytest.mark.asyncio
    async def test_save_and_load_clusters(self, workspace):
        """测试保存和加载聚类"""
        cluster = ClusterResult(
            cluster_id=1,
            label="Test Cluster",
            description="A test cluster",
            paper_ids=["paper1", "paper2"],
            representative_papers=[],
            sub_themes=["theme1", "theme2"],
            size=2,
        )

        await workspace.save_clusters([cluster])

        clusters = await workspace.get_clusters()
        assert len(clusters) == 1
        assert clusters[0].cluster_id == 1
        assert clusters[0].label == "Test Cluster"

    @pytest.mark.asyncio
    async def test_save_and_get_summary(self, workspace):
        """测试保存和获取摘要"""
        await workspace.save_summary("test_key", "This is a test summary")

        summary = await workspace.get_summary("test_key")
        assert summary == "This is a test summary"

    @pytest.mark.asyncio
    async def test_save_and_get_embeddings(self, workspace):
        """测试保存和获取embeddings"""
        embedding = [0.1, 0.2, 0.3] * 100  # 300维向量
        await workspace.save_embedding("paper1", embedding)

        retrieved = await workspace.get_embedding("paper1")
        assert retrieved == embedding

    @pytest.mark.asyncio
    async def test_checkpoint(self, workspace, sample_papers):
        """测试检查点功能"""
        # 添加一些数据
        records = [
            LiteratureRecord(
                id=p["paperId"],
                title=p["title"],
                authors=[a["name"] for a in p["authors"]],
                abstract=p["abstract"],
                year=p["year"],
                source=p["source"],
                url=p["url"],
            )
            for p in sample_papers[:3]
        ]
        await workspace.add_literature(records)

        # 创建检查点
        checkpoint_path = await workspace.create_checkpoint("test_checkpoint")
        assert checkpoint_path is not None
        assert "test_checkpoint" in checkpoint_path

        # 检查检查点目录
        checkpoint_dir = Path(checkpoint_path)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "literature").exists()

    @pytest.mark.asyncio
    async def test_workspace_info(self, workspace):
        """测试获取工作区信息"""
        info = workspace.get_workspace_info()
        assert "literature_count" in info
        assert "cluster_count" in info
        assert info["literature_count"] == 0
