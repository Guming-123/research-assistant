"""
Shared Workspace for Multi-Agent Literature Review System
共享工作区 - 所有Agent共享的数据存储和管理
"""

import asyncio
import json
import os
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import hashlib

from langchain_core.documents import Document
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class WorkspaceEntry:
    """工作区条目"""

    key: str
    value: Any
    agent: str
    stage: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "key": self.key,
            "agent": self.agent,
            "stage": self.stage,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "value_type": type(self.value).__name__,
        }


@dataclass
class LiteratureRecord:
    """文献记录"""

    id: str
    title: str
    authors: List[str]
    abstract: str
    year: int
    source: str  # 数据库来源
    url: str
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    venue: Optional[str] = None  # 会议/期刊名称
    citation_count: Optional[int] = None
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "title": self.title,
            "authors": self.authors,
            "abstract": self.abstract,
            "year": self.year,
            "source": self.source,
            "url": self.url,
            "doi": self.doi,
            "keywords": self.keywords,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "pdf_path": self.pdf_path,
            "relevance_score": self.relevance_score,
            "cluster_id": self.cluster_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LiteratureRecord":
        """从字典创建实例"""
        return cls(**data)

    def __hash__(self) -> int:
        """用于去重的哈希值"""
        content = f"{self.title}{self.authors}{self.year}".lower()
        return hashlib.md5(content.encode()).hexdigest()


@dataclass
class ClusterResult:
    """聚类结果"""

    cluster_id: int
    label: str
    description: str
    paper_ids: List[str]
    representative_papers: List[Dict[str, Any]]
    sub_themes: List[str]
    size: int
    silhouette_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "description": self.description,
            "paper_ids": self.paper_ids,
            "representative_papers": self.representative_papers,
            "sub_themes": self.sub_themes,
            "size": self.size,
            "silhouette_score": self.silhouette_score,
            "metadata": self.metadata,
        }


class SharedWorkspace:
    """
    共享工作区

    功能：
    - 存储和管理文献数据
    - 存储中间结果和元数据
    - 支持Agent间数据共享
    - 持久化存储
    - 版本管理
    """

    def __init__(self, base_path: str = "./workspace"):
        """
        初始化工作区

        Args:
            base_path: 工作区基础路径
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # 数据存储
        self._literature: Dict[str, LiteratureRecord] = {}
        self._clusters: Dict[int, ClusterResult] = {}
        self._summaries: Dict[str, str] = {}
        self._embeddings: Dict[str, List[float]] = {}
        self._metadata: Dict[str, Any] = {}

        # 创建子目录
        self._create_directories()

    def _create_directories(self) -> None:
        """创建必要的目录结构"""
        dirs = [
            "literature",
            "clusters",
            "summaries",
            "embeddings",
            "reports",
            "checkpoints",
            "pdfs",
        ]
        for d in dirs:
            (self.base_path / d).mkdir(parents=True, exist_ok=True)

    # ========== 文献管理 ==========

    async def add_literature(
        self, records: Union[LiteratureRecord, List[LiteratureRecord]]
    ) -> int:
        """
        添加文献记录

        Args:
            records: 单个或多个文献记录

        Returns:
            添加的数量
        """
        if isinstance(records, LiteratureRecord):
            records = [records]

        count = 0
        for record in records:
            # 使用title+authors+year的hash作为唯一ID
            if not record.id:
                record.id = str(hash(record))

            # 去重检查
            if record.id not in self._literature:
                self._literature[record.id] = record
                count += 1

        await self._save_literature()
        return count

    async def get_literature(
        self,
        paper_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[LiteratureRecord]:
        """
        获取文献记录

        Args:
            paper_ids: 文献ID列表
            filters: 过滤条件

        Returns:
            文献记录列表
        """
        if paper_ids:
            return [self._literature[pid] for pid in paper_ids if pid in self._literature]

        records = list(self._literature.values())

        if filters:
            filtered = []
            for record in records:
                match = True
                for key, value in filters.items():
                    if getattr(record, key, None) != value:
                        match = False
                        break
                if match:
                    filtered.append(record)
            return filtered

        return records

    async def update_literature(self, paper_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新文献记录

        Args:
            paper_id: 文献ID
            updates: 更新字段

        Returns:
            是否成功
        """
        if paper_id not in self._literature:
            return False

        for key, value in updates.items():
            if hasattr(self._literature[paper_id], key):
                setattr(self._literature[paper_id], key, value)

        await self._save_literature()
        return True

    async def remove_literature(self, paper_ids: List[str]) -> int:
        """
        移除文献记录

        Args:
            paper_ids: 文献ID列表

        Returns:
            移除的数量
        """
        count = 0
        for pid in paper_ids:
            if pid in self._literature:
                del self._literature[pid]
                count += 1

        await self._save_literature()
        return count

    def get_literature_count(self) -> int:
        """获取文献总数"""
        return len(self._literature)

    # ========== 聚类结果管理 ==========

    async def save_clusters(self, clusters: List[ClusterResult]) -> None:
        """保存聚类结果"""
        for cluster in clusters:
            self._clusters[cluster.cluster_id] = cluster
        await self._save_clusters()

    async def get_clusters(self) -> List[ClusterResult]:
        """获取所有聚类结果"""
        return list(self._clusters.values())

    async def get_cluster(self, cluster_id: int) -> Optional[ClusterResult]:
        """获取单个聚类"""
        return self._clusters.get(cluster_id)

    async def get_cluster_papers(self, cluster_id: int) -> List[LiteratureRecord]:
        """获取聚类中的所有文献"""
        cluster = await self.get_cluster(cluster_id)
        if not cluster:
            return []
        return await self.get_literature(paper_ids=cluster.paper_ids)

    # ========== 摘要管理 ==========

    async def save_summary(self, key: str, summary: str) -> None:
        """保存摘要"""
        self._summaries[key] = summary
        await self._save_summaries()

    async def get_summary(self, key: str) -> Optional[str]:
        """获取摘要"""
        return self._summaries.get(key)

    async def get_all_summaries(self) -> Dict[str, str]:
        """获取所有摘要"""
        return self._summaries.copy()

    # ========== Embedding管理 ==========

    async def save_embedding(self, paper_id: str, embedding: List[float]) -> None:
        """保存embedding"""
        self._embeddings[paper_id] = embedding
        await self._save_embeddings()

    async def get_embedding(self, paper_id: str) -> Optional[List[float]]:
        """获取embedding"""
        return self._embeddings.get(paper_id)

    async def get_all_embeddings(self) -> Dict[str, List[float]]:
        """获取所有embeddings"""
        return self._embeddings.copy()

    # ========== 通用存储 ==========

    async def save(
        self, key: str, value: Any, agent: str = "system", stage: str = "intermediate"
    ) -> None:
        """
        通用存储方法

        Args:
            key: 存储键
            value: 存储值
            agent: Agent名称
            stage: 阶段标识
        """
        entry = WorkspaceEntry(key=key, value=value, agent=agent, stage=stage)

        # 根据类型选择存储方式
        if isinstance(value, LiteratureRecord):
            await self.add_literature(value)
        elif isinstance(value, list) and value and isinstance(value[0], LiteratureRecord):
            await self.add_literature(value)
        elif isinstance(value, ClusterResult):
            self._clusters[key] = value
        elif isinstance(value, str) and "summary" in key.lower():
            await self.save_summary(key, value)
        else:
            # 默认存储为JSON
            self._metadata[key] = {
                "value": value,
                "agent": agent,
                "stage": stage,
                "timestamp": datetime.now().isoformat(),
            }

    async def load(self, key: str) -> Optional[Any]:
        """通用加载方法"""
        # 从各个存储区域查找
        if key in self._literature:
            return self._literature[key]
        if key in self._clusters:
            return self._clusters[key]
        if key in self._summaries:
            return self._summaries[key]
        if key in self._metadata:
            return self._metadata[key].get("value")
        return None

    # ========== 持久化 ==========

    async def _save_literature(self) -> None:
        """保存文献数据"""
        path = self.base_path / "literature" / "records.json"
        data = {pid: record.to_dict() for pid, record in self._literature.items()}
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))

    async def _save_clusters(self) -> None:
        """保存聚类数据"""
        path = self.base_path / "clusters" / "results.json"
        data = {str(cid): c.to_dict() for cid, c in self._clusters.items()}
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(data, ensure_ascii=False, indent=2))

    async def _save_summaries(self) -> None:
        """保存摘要数据"""
        path = self.base_path / "summaries" / "all_summaries.json"
        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(self._summaries, ensure_ascii=False, indent=2))

    async def _save_embeddings(self) -> None:
        """保存embedding数据（使用JSON而非pickle）"""
        path = self.base_path / "embeddings" / "embeddings.json"

        # 转换为 JSON 可序列化的格式
        embeddings_serializable = {
            k: v for k, v in self._embeddings.items()
        }

        async with aiofiles.open(path, "w", encoding="utf-8") as f:
            await f.write(json.dumps(embeddings_serializable, ensure_ascii=False, indent=2))

    async def load_all(self) -> None:
        """加载所有持久化数据"""
        # 加载文献
        lit_path = self.base_path / "literature" / "records.json"
        if lit_path.exists():
            async with aiofiles.open(lit_path, encoding="utf-8") as f:
                data = json.loads(await f.read())
                self._literature = {
                    pid: LiteratureRecord.from_dict(rec) for pid, rec in data.items()
                }

        # 加载聚类
        clust_path = self.base_path / "clusters" / "results.json"
        if clust_path.exists():
            async with aiofiles.open(clust_path, encoding="utf-8") as f:
                data = json.loads(await f.read())
                self._clusters = {
                    int(cid): ClusterResult(**rec) for cid, rec in data.items()
                }

        # 加载摘要
        summ_path = self.base_path / "summaries" / "all_summaries.json"
        if summ_path.exists():
            async with aiofiles.open(summ_path, encoding="utf-8") as f:
                self._summaries = json.loads(await f.read())

        # 加载 embeddings
        emb_path = self.base_path / "embeddings" / "embeddings.json"
        if emb_path.exists():
            try:
                async with aiofiles.open(emb_path, encoding="utf-8") as f:
                    data = json.loads(await f.read())
                    self._embeddings = {k: v for k, v in data.items()}
            except Exception as e:
                logger.warning(f"Failed to load embeddings: {e}")
                self._embeddings = {}

    async def create_checkpoint(self, name: str) -> str:
        """
        创建检查点

        Args:
            name: 检查点名称

        Returns:
            检查点路径
        """
        checkpoint_dir = self.base_path / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 复制所有数据到检查点目录
        import shutil

        for d in ["literature", "clusters", "summaries", "embeddings"]:
            src = self.base_path / d
            if src.exists():
                dst = checkpoint_dir / d
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        # 保存元数据
        meta_path = checkpoint_dir / "metadata.json"
        async with aiofiles.open(meta_path, "w", encoding="utf-8") as f:
            await f.write(
                json.dumps(
                    {"name": name, "timestamp": datetime.now().isoformat()}, ensure_ascii=False
                )
            )

        return str(checkpoint_dir)

    async def restore_checkpoint(self, name: str) -> bool:
        """
        恢复检查点

        Args:
            name: 检查点名称

        Returns:
            是否成功
        """
        checkpoint_dir = self.base_path / "checkpoints" / name
        if not checkpoint_dir.exists():
            return False

        # 恢复数据
        import shutil

        for d in ["literature", "clusters", "summaries", "embeddings"]:
            src = checkpoint_dir / d
            if src.exists():
                dst = self.base_path / d
                if dst.exists():
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)

        await self.load_all()
        return True

    def get_workspace_info(self) -> Dict[str, Any]:
        """获取工作区摘要信息"""
        return {
            "literature_count": len(self._literature),
            "cluster_count": len(self._clusters),
            "summary_count": len(self._summaries),
            "embedding_count": len(self._embeddings),
            "base_path": str(self.base_path),
        }
