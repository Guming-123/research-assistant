"""
Shared Workspace for Multi-Agent Literature Review System
共享工作区 - SQLite 持久化存储，论文全局去重 + research_topic 派生数据隔离
"""

import asyncio
import json
import logging
import shutil
import sqlite3
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════
# 数据类（保持不变）
# ════════════════════════════════════════════

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
    source: str
    url: str
    doi: Optional[str] = None
    keywords: List[str] = field(default_factory=list)
    venue: Optional[str] = None
    citation_count: Optional[int] = None
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    embedding: Optional[List[float]] = None
    relevance_score: Optional[float] = None
    cluster_id: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
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
        return cls(**data)

    def __hash__(self) -> int:
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


# ════════════════════════════════════════════
# SQLite 共享工作区 — 论文全局共享 + research_topic 派生隔离
# ════════════════════════════════════════════

_SCHEMA = """
CREATE TABLE IF NOT EXISTS literature (
    id              TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    authors         TEXT NOT NULL,
    abstract        TEXT,
    year            INTEGER,
    source          TEXT,
    url             TEXT,
    doi             TEXT,
    keywords        TEXT NOT NULL DEFAULT '[]',
    venue           TEXT,
    citation_count  INTEGER,
    pdf_path        TEXT,
    full_text       TEXT,
    relevance_score REAL,
    cluster_id      INTEGER,
    metadata        TEXT NOT NULL DEFAULT '{}',
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS embeddings (
    paper_id    TEXT PRIMARY KEY,
    embedding   BLOB NOT NULL,
    dimensions  INTEGER NOT NULL DEFAULT 768,
    FOREIGN KEY (paper_id) REFERENCES literature(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS clusters (
    research_topic        TEXT NOT NULL,
    cluster_id            INTEGER NOT NULL,
    label                 TEXT NOT NULL DEFAULT '',
    description           TEXT NOT NULL DEFAULT '',
    paper_ids             TEXT NOT NULL DEFAULT '[]',
    representative_papers TEXT NOT NULL DEFAULT '[]',
    sub_themes            TEXT NOT NULL DEFAULT '[]',
    size                  INTEGER NOT NULL DEFAULT 0,
    silhouette_score      REAL,
    metadata              TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (research_topic, cluster_id)
);
CREATE TABLE IF NOT EXISTS summaries (
    research_topic TEXT NOT NULL,
    key            TEXT NOT NULL,
    summary        TEXT NOT NULL,
    created_at     TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at     TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (research_topic, key)
);
CREATE TABLE IF NOT EXISTS workspace_metadata (
    research_topic TEXT NOT NULL,
    key            TEXT NOT NULL,
    value          TEXT NOT NULL,
    agent          TEXT NOT NULL DEFAULT 'system',
    stage          TEXT NOT NULL DEFAULT 'intermediate',
    timestamp      TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (research_topic, key)
);
CREATE INDEX IF NOT EXISTS idx_clust_topic ON clusters(research_topic);
CREATE INDEX IF NOT EXISTS idx_summ_topic ON summaries(research_topic);
CREATE INDEX IF NOT EXISTS idx_meta_topic ON workspace_metadata(research_topic);
"""

_LIT_COLUMNS = [
    "id", "title", "authors", "abstract", "year", "source", "url",
    "doi", "keywords", "venue", "citation_count", "pdf_path",
    "full_text", "relevance_score", "cluster_id", "metadata",
]

# 用于旧 schema 的列（含 topic_id），迁移时使用
_LIT_COLUMNS_OLD = [
    "id", "title", "authors", "abstract", "year", "source", "url",
    "doi", "keywords", "venue", "citation_count", "pdf_path",
    "full_text", "relevance_score", "cluster_id", "metadata",
]


class SharedWorkspace:
    """
    基于 SQLite 的共享工作区。

    - literature / embeddings: 全局共享，按 id 去重复用
    - clusters / summaries / workspace_metadata: 按 research_topic 隔离
    """

    def __init__(self, base_path: str = "./workspace"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        for d in ["reports", "checkpoints", "pdfs"]:
            (self.base_path / d).mkdir(parents=True, exist_ok=True)

        db_path = self.base_path / "research.db"
        self._migrate_schema(db_path)
        self._migrate_from_json(db_path)

        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(_SCHEMA)

        # 当前 topic 上下文
        self._current_topic: Optional[str] = None

    def close(self):
        if self._conn:
            self._conn.close()

    @property
    def research_topic(self) -> str:
        """当前 research_topic，未设置时返回空字符串"""
        return self._current_topic or ""

    @property
    def topic(self) -> Optional[str]:
        return self._current_topic

    def set_topic(self, topic: str) -> str:
        """设置当前研究主题，返回 research_topic"""
        self._current_topic = topic
        logger.info(f"Workspace topic set: '{topic}'")
        return topic

    # ──────────── 辅助方法 ────────────

    @staticmethod
    def _emb_to_blob(vec: List[float]) -> bytes:
        return np.array(vec, dtype=np.float32).tobytes()

    @staticmethod
    def _blob_to_emb(blob: bytes) -> List[float]:
        return np.frombuffer(blob, dtype=np.float32).tolist()

    @staticmethod
    def _row_to_literature(row: sqlite3.Row) -> LiteratureRecord:
        d = dict(row)
        d["authors"] = json.loads(d["authors"]) if isinstance(d["authors"], str) else d["authors"]
        d["keywords"] = json.loads(d["keywords"]) if isinstance(d["keywords"], str) else (d["keywords"] or [])
        d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else (d["metadata"] or {})
        d.pop("created_at", None)
        d.pop("updated_at", None)
        d.pop("topic_id", None)
        d.pop("research_topic", None)
        d.pop("embedding", None)
        return LiteratureRecord.from_dict(d)

    @staticmethod
    def _literature_to_params(rec: LiteratureRecord) -> tuple:
        return (
            rec.id,
            rec.title,
            json.dumps(rec.authors, ensure_ascii=False),
            rec.abstract,
            rec.year,
            rec.source,
            rec.url,
            rec.doi,
            json.dumps(rec.keywords or [], ensure_ascii=False),
            rec.venue,
            rec.citation_count,
            rec.pdf_path,
            rec.full_text,
            rec.relevance_score,
            rec.cluster_id,
            json.dumps(rec.metadata or {}, ensure_ascii=False),
        )

    @staticmethod
    def _row_to_cluster(row: sqlite3.Row) -> ClusterResult:
        d = dict(row)
        d["paper_ids"] = json.loads(d["paper_ids"]) if isinstance(d["paper_ids"], str) else (d["paper_ids"] or [])
        d["representative_papers"] = json.loads(d["representative_papers"]) if isinstance(d["representative_papers"], str) else (d["representative_papers"] or [])
        d["sub_themes"] = json.loads(d["sub_themes"]) if isinstance(d["sub_themes"], str) else (d["sub_themes"] or [])
        d["metadata"] = json.loads(d["metadata"]) if isinstance(d["metadata"], str) else (d["metadata"] or {})
        d.pop("topic_id", None)
        d.pop("research_topic", None)
        return ClusterResult(**d)

    # ──────────── Schema 迁移 ────────────

    def _migrate_schema(self, db_path: Path) -> None:
        """从旧 schema（topic_id 隔离）迁移到新 schema（论文全局共享）"""
        if not db_path.exists():
            return

        conn = sqlite3.connect(str(db_path))
        try:
            # 检查是否是旧 schema（literature 表有 topic_id 列）
            cols = [r[1] for r in conn.execute("PRAGMA table_info(literature)").fetchall()]
            if "topic_id" not in cols:
                return  # 已经是新 schema

            logger.info("正在从旧 schema (topic_id) 迁移到新 schema (全局论文 + research_topic)...")

            with conn:
                # 获取所有旧 topic_id 及其对应的 research_topic
                # 旧数据中无法还原原始 topic 字符串，用 topic_id 作为 fallback
                old_topics = [r[0] for r in conn.execute(
                    "SELECT DISTINCT topic_id FROM literature"
                ).fetchall()]

                # 创建临时表
                conn.executescript("""
                    ALTER TABLE literature RENAME TO literature_old;
                    ALTER TABLE embeddings RENAME TO embeddings_old;
                    ALTER TABLE clusters RENAME TO clusters_old;
                    ALTER TABLE summaries RENAME TO summaries_old;
                    ALTER TABLE workspace_metadata RENAME TO workspace_metadata_old;
                """)

                # 创建新 schema
                conn.executescript(_SCHEMA)

                # 迁移 literature: 去掉 topic_id，全局去重
                conn.execute("""
                    INSERT OR IGNORE INTO literature
                    (id, title, authors, abstract, year, source, url, doi, keywords,
                     venue, citation_count, pdf_path, full_text, relevance_score,
                     cluster_id, metadata, created_at, updated_at)
                    SELECT
                    id, title, authors, abstract, year, source, url, doi, keywords,
                     venue, citation_count, pdf_path, full_text, relevance_score,
                     cluster_id, metadata, created_at, updated_at
                    FROM literature_old
                """)

                # 迁移 embeddings: 去掉 topic_id
                conn.execute("""
                    INSERT OR IGNORE INTO embeddings (paper_id, embedding, dimensions)
                    SELECT paper_id, embedding, dimensions FROM embeddings_old
                """)

                # 迁移 clusters/summaries/metadata: topic_id → research_topic
                for old_tid in old_topics:
                    conn.execute("""
                        INSERT OR IGNORE INTO clusters
                        (research_topic, cluster_id, label, description, paper_ids,
                         representative_papers, sub_themes, size, silhouette_score, metadata)
                        SELECT ?, cluster_id, label, description, paper_ids,
                         representative_papers, sub_themes, size, silhouette_score, metadata
                        FROM clusters_old WHERE topic_id = ?
                    """, (old_tid, old_tid))

                    conn.execute("""
                        INSERT OR IGNORE INTO summaries (research_topic, key, summary, created_at, updated_at)
                        SELECT ?, key, summary, created_at, updated_at
                        FROM summaries_old WHERE topic_id = ?
                    """, (old_tid, old_tid))

                    conn.execute("""
                        INSERT OR IGNORE INTO workspace_metadata (research_topic, key, value, agent, stage, timestamp)
                        SELECT ?, key, value, agent, stage, timestamp
                        FROM workspace_metadata_old WHERE topic_id = ?
                    """, (old_tid, old_tid))

                # 删除旧表
                conn.executescript("""
                    DROP TABLE literature_old;
                    DROP TABLE embeddings_old;
                    DROP TABLE clusters_old;
                    DROP TABLE summaries_old;
                    DROP TABLE workspace_metadata_old;
                """)

                # 删除旧索引
                conn.execute("DROP INDEX IF EXISTS idx_lit_topic")
                conn.execute("DROP INDEX IF EXISTS idx_emb_topic")
                conn.execute("DROP INDEX IF EXISTS idx_clust_topic")

            logger.info("Schema 迁移完成: topic_id → 全局论文 + research_topic")
        except Exception as e:
            logger.error(f"Schema 迁移失败: {e}")
            conn.close()
            raise

    # ──────────── JSON 迁移 ────────────

    def _migrate_from_json(self, db_path: Path) -> None:
        if db_path.exists():
            return
        lit_path = self.base_path / "literature" / "records.json"
        if not lit_path.exists():
            return

        logger.info("正在从 JSON 迁移到 SQLite...")
        rq_path = self.base_path / "rq_tree.json"
        if rq_path.exists():
            rq_data = json.loads(rq_path.read_text(encoding="utf-8"))
            old_topic = rq_data.get("research_topic", "default")
        else:
            old_topic = "default"

        conn = sqlite3.connect(str(db_path))
        conn.executescript(_SCHEMA)
        try:
            with conn:
                data = json.loads(lit_path.read_text(encoding="utf-8"))
                for pid, rec in data.items():
                    rec.setdefault("id", pid)
                    lr = LiteratureRecord.from_dict(rec)
                    conn.execute(
                        f"INSERT OR IGNORE INTO literature ({','.join(_LIT_COLUMNS)}) VALUES ({','.join(['?']*len(_LIT_COLUMNS))})",
                        [json.dumps(getattr(lr, c), ensure_ascii=False) if c in ("authors", "keywords", "metadata") else getattr(lr, c) for c in _LIT_COLUMNS],
                    )
                clust_path = self.base_path / "clusters" / "results.json"
                if clust_path.exists():
                    cdata = json.loads(clust_path.read_text(encoding="utf-8"))
                    for cid, c in cdata.items():
                        conn.execute(
                            "INSERT OR REPLACE INTO clusters (research_topic,cluster_id,label,description,paper_ids,representative_papers,sub_themes,size,silhouette_score,metadata) VALUES (?,?,?,?,?,?,?,?,?,?)",
                            (old_topic, c.get("cluster_id", int(cid)), c.get("label", ""), c.get("description", ""),
                             json.dumps(c.get("paper_ids", []), ensure_ascii=False),
                             json.dumps(c.get("representative_papers", []), ensure_ascii=False),
                             json.dumps(c.get("sub_themes", []), ensure_ascii=False),
                             c.get("size", 0), c.get("silhouette_score"),
                             json.dumps(c.get("metadata", {}), ensure_ascii=False)),
                        )
                summ_path = self.base_path / "summaries" / "all_summaries.json"
                if summ_path.exists():
                    sdata = json.loads(summ_path.read_text(encoding="utf-8"))
                    for k, v in sdata.items():
                        conn.execute("INSERT OR REPLACE INTO summaries (research_topic, key, summary) VALUES (?, ?, ?)", (old_topic, k, v))
                emb_path = self.base_path / "embeddings" / "embeddings.json"
                if emb_path.exists():
                    edata = json.loads(emb_path.read_text(encoding="utf-8"))
                    for pid, vec in edata.items():
                        conn.execute(
                            "INSERT OR REPLACE INTO embeddings (paper_id, embedding, dimensions) VALUES (?, ?, ?)",
                            (pid, self._emb_to_blob(vec), len(vec)),
                        )
            logger.info(f"JSON → SQLite 迁移完成，旧数据归入 research_topic={old_topic}")
        except Exception as e:
            logger.error(f"迁移失败: {e}")
            conn.close()
            db_path.unlink(missing_ok=True)
            raise

    # ════════════════════════════════════════
    # 文献管理（全局共享）
    # ════════════════════════════════════════

    async def add_literature(
        self, records: Union[LiteratureRecord, List[LiteratureRecord]]
    ) -> int:
        if isinstance(records, LiteratureRecord):
            records = [records]

        def _insert():
            count = 0
            with self._conn:
                for rec in records:
                    if not rec.id:
                        rec.id = str(hash(rec))
                    try:
                        self._conn.execute(
                            f"INSERT OR IGNORE INTO literature ({','.join(_LIT_COLUMNS)}) VALUES ({','.join(['?']*len(_LIT_COLUMNS))})",
                            [json.dumps(getattr(rec, c), ensure_ascii=False) if c in ("authors", "keywords", "metadata") else getattr(rec, c) for c in _LIT_COLUMNS],
                        )
                        count += self._conn.changes
                    except Exception:
                        pass
            return count

        return await asyncio.to_thread(_insert)

    async def get_literature(
        self,
        paper_ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[LiteratureRecord]:

        def _query():
            if paper_ids:
                ph = ",".join(["?"] * len(paper_ids))
                rows = self._conn.execute(
                    f"SELECT * FROM literature WHERE id IN ({ph})",
                    paper_ids,
                ).fetchall()
            elif filters:
                conds, vals = [], []
                for k, v in filters.items():
                    if k in _LIT_COLUMNS:
                        conds.append(f"{k} = ?")
                        vals.append(v)
                rows = self._conn.execute(
                    f"SELECT * FROM literature WHERE {' AND '.join(conds)}", vals
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM literature"
                ).fetchall()
            return [self._row_to_literature(r) for r in rows]

        return await asyncio.to_thread(_query)

    async def update_literature(self, paper_id: str, updates: Dict[str, Any]) -> bool:

        def _update():
            sets, vals = [], []
            json_fields = {"authors", "keywords", "metadata"}
            for k, v in updates.items():
                if k in _LIT_COLUMNS and k != "id":
                    if k in json_fields:
                        v = json.dumps(v, ensure_ascii=False)
                    sets.append(f"{k} = ?")
                    vals.append(v)
            if not sets:
                return False
            sets.append("updated_at = datetime('now')")
            vals.append(paper_id)
            with self._conn:
                self._conn.execute(
                    f"UPDATE literature SET {','.join(sets)} WHERE id = ?", vals
                )
            return self._conn.changes > 0

        return await asyncio.to_thread(_update)

    async def remove_literature(self, paper_ids: List[str]) -> int:

        def _delete():
            ph = ",".join(["?"] * len(paper_ids))
            with self._conn:
                self._conn.execute(
                    f"DELETE FROM literature WHERE id IN ({ph})",
                    paper_ids,
                )
            return self._conn.changes

        return await asyncio.to_thread(_delete)

    def get_literature_count(self) -> int:
        row = self._conn.execute(
            "SELECT COUNT(*) FROM literature"
        ).fetchone()
        return row[0]

    # ════════════════════════════════════════
    # 聚类管理（按 research_topic 隔离）
    # ════════════════════════════════════════

    async def save_clusters(self, clusters: List[ClusterResult]) -> None:
        rt = self.research_topic

        def _save():
            with self._conn:
                self._conn.execute("DELETE FROM clusters WHERE research_topic = ?", (rt,))
                for c in clusters:
                    self._conn.execute(
                        "INSERT INTO clusters (research_topic,cluster_id,label,description,paper_ids,representative_papers,sub_themes,size,silhouette_score,metadata) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (rt, c.cluster_id, c.label, c.description,
                         json.dumps(c.paper_ids, ensure_ascii=False),
                         json.dumps(c.representative_papers, ensure_ascii=False),
                         json.dumps(c.sub_themes, ensure_ascii=False),
                         c.size, c.silhouette_score,
                         json.dumps(c.metadata, ensure_ascii=False)),
                    )

        await asyncio.to_thread(_save)

    async def get_clusters(self) -> List[ClusterResult]:
        rt = self.research_topic

        def _query():
            rows = self._conn.execute(
                "SELECT * FROM clusters WHERE research_topic = ?", (rt,)
            ).fetchall()
            return [self._row_to_cluster(r) for r in rows]

        return await asyncio.to_thread(_query)

    async def get_cluster(self, cluster_id: int) -> Optional[ClusterResult]:
        rt = self.research_topic

        def _query():
            row = self._conn.execute(
                "SELECT * FROM clusters WHERE research_topic = ? AND cluster_id = ?",
                (rt, cluster_id),
            ).fetchone()
            return self._row_to_cluster(row) if row else None

        return await asyncio.to_thread(_query)

    async def get_cluster_papers(self, cluster_id: int) -> List[LiteratureRecord]:
        cluster = await self.get_cluster(cluster_id)
        if not cluster:
            return []
        return await self.get_literature(paper_ids=cluster.paper_ids)

    # ════════════════════════════════════════
    # 摘要管理（按 research_topic 隔离）
    # ════════════════════════════════════════

    async def save_summary(self, key: str, summary: str) -> None:
        rt = self.research_topic

        def _save():
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO summaries (research_topic, key, summary, updated_at) VALUES (?, ?, ?, datetime('now'))",
                    (rt, key, summary),
                )

        await asyncio.to_thread(_save)

    async def get_summary(self, key: str) -> Optional[str]:
        rt = self.research_topic

        def _query():
            row = self._conn.execute(
                "SELECT summary FROM summaries WHERE research_topic = ? AND key = ?", (rt, key)
            ).fetchone()
            return row[0] if row else None

        return await asyncio.to_thread(_query)

    async def get_all_summaries(self) -> Dict[str, str]:
        rt = self.research_topic

        def _query():
            rows = self._conn.execute(
                "SELECT key, summary FROM summaries WHERE research_topic = ?", (rt,)
            ).fetchall()
            return {r[0]: r[1] for r in rows}

        return await asyncio.to_thread(_query)

    # ════════════════════════════════════════
    # Embedding 管理（全局共享）
    # ════════════════════════════════════════

    async def save_embedding(self, paper_id: str, embedding: List[float]) -> None:
        await self.batch_save_embeddings({paper_id: embedding})

    async def batch_save_embeddings(self, embeddings: Dict[str, List[float]]) -> None:

        def _save():
            with self._conn:
                for pid, vec in embeddings.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO embeddings (paper_id, embedding, dimensions) VALUES (?, ?, ?)",
                        (pid, self._emb_to_blob(vec), len(vec)),
                    )

        await asyncio.to_thread(_save)

    async def get_embedding(self, paper_id: str) -> Optional[List[float]]:

        def _query():
            row = self._conn.execute(
                "SELECT embedding FROM embeddings WHERE paper_id = ?",
                (paper_id,),
            ).fetchone()
            return self._blob_to_emb(row[0]) if row else None

        return await asyncio.to_thread(_query)

    async def get_all_embeddings(self) -> Dict[str, List[float]]:

        def _query():
            rows = self._conn.execute(
                "SELECT paper_id, embedding FROM embeddings"
            ).fetchall()
            return {r[0]: self._blob_to_emb(r[1]) for r in rows}

        return await asyncio.to_thread(_query)

    # ════════════════════════════════════════
    # 批量操作 & 便捷方法
    # ════════════════════════════════════════

    async def batch_update_relevance(self, updates: Dict[str, Optional[float]]) -> None:
        """批量更新相关度分数。论文全局共享，不同主题对同一论文的评分会互相覆盖。"""

        def _update():
            with self._conn:
                for pid, score in updates.items():
                    self._conn.execute(
                        "UPDATE literature SET relevance_score = ?, updated_at = datetime('now') WHERE id = ?",
                        (score, pid),
                    )

        await asyncio.to_thread(_update)

    async def reset_all_relevance_scores(self) -> None:
        """将所有论文的 relevance_score 重置为 NULL（仅在无搜索记录时的安全回退）"""

        def _reset():
            with self._conn:
                self._conn.execute("UPDATE literature SET relevance_score = NULL")

        await asyncio.to_thread(_reset)

    async def clear_topic(self) -> None:
        """清除当前主题的派生数据，保留论文和 embeddings"""
        rt = self.research_topic

        def _clear():
            with self._conn:
                for table in ["clusters", "summaries", "workspace_metadata"]:
                    self._conn.execute(f"DELETE FROM {table} WHERE research_topic = ?", (rt,))
                # 注意：不再全局清除 relevance_score
                # Screen Agent 会仅对当前搜索的论文设置 relevance_score，
                # 避免清除旧主题论文的已有评分

        await asyncio.to_thread(_clear)
        logger.info(f"Cleared derived data for topic: '{rt}' (literature & embeddings preserved)")

    async def clear_all(self) -> None:
        """清除所有数据"""
        def _clear():
            with self._conn:
                for table in ["embeddings", "clusters", "summaries", "workspace_metadata", "literature"]:
                    self._conn.execute(f"DELETE FROM {table}")

        await asyncio.to_thread(_clear)

    async def save_metadata_item(self, key: str, value: Any,
                                  agent: str = "system", stage: str = "intermediate") -> None:
        rt = self.research_topic

        def _save():
            with self._conn:
                self._conn.execute(
                    "INSERT OR REPLACE INTO workspace_metadata (research_topic, key, value, agent, stage, timestamp) VALUES (?, ?, ?, ?, ?, datetime('now'))",
                    (rt, key, json.dumps(value, ensure_ascii=False), agent, stage),
                )

        await asyncio.to_thread(_save)

    async def get_metadata_item(self, key: str) -> Optional[Any]:
        rt = self.research_topic

        def _query():
            row = self._conn.execute(
                "SELECT value FROM workspace_metadata WHERE research_topic = ? AND key = ?",
                (rt, key),
            ).fetchone()
            return json.loads(row[0]) if row else None

        return await asyncio.to_thread(_query)

    async def get_literature_as_dicts(self, paper_ids: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        records = await self.get_literature(paper_ids=paper_ids)
        return [r.to_dict() for r in records]

    async def get_clusters_as_dicts(self) -> Dict[str, Dict[str, Any]]:
        clusters = await self.get_clusters()
        return {str(c.cluster_id): c.to_dict() for c in clusters}

    async def get_embeddings_as_dict(self) -> Dict[str, List[float]]:
        return await self.get_all_embeddings()

    # ════════════════════════════════════════
    # 多 topic 查询
    # ════════════════════════════════════════

    def list_topics(self) -> List[Dict[str, Any]]:
        """列出数据库中所有 research_topic 及其统计"""
        rows = self._conn.execute("""
            SELECT c.research_topic,
                   COUNT(DISTINCT c.cluster_id) as cluster_count,
                   (SELECT COUNT(*) FROM summaries s WHERE s.research_topic = c.research_topic) as summary_count
            FROM clusters c
            GROUP BY c.research_topic
            ORDER BY cluster_count DESC
        """).fetchall()
        return [dict(r) for r in rows]

    # ════════════════════════════════════════
    # 通用存储
    # ════════════════════════════════════════

    async def save(self, key: str, value: Any, agent: str = "system", stage: str = "intermediate") -> None:
        if isinstance(value, LiteratureRecord):
            await self.add_literature(value)
        elif isinstance(value, list) and value and isinstance(value[0], LiteratureRecord):
            await self.add_literature(value)
        elif isinstance(value, str) and "summary" in key.lower():
            await self.save_summary(key, value)
        else:
            await self.save_metadata_item(key, value, agent, stage)

    async def load(self, key: str) -> Optional[Any]:
        summary = await self.get_summary(key)
        if summary is not None:
            return summary
        meta = await self.get_metadata_item(key)
        if meta is not None:
            return meta
        lit = await self.get_literature(paper_ids=[key])
        if lit:
            return lit[0]
        return None

    # ════════════════════════════════════════
    # 持久化 & 检查点
    # ════════════════════════════════════════

    async def load_all(self) -> None:
        """SQLite 模式下数据已在磁盘，此方法为兼容保留"""
        pass

    async def create_checkpoint(self, name: str) -> str:
        checkpoint_dir = self.base_path / "checkpoints" / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        def _copy():
            db_src = self.base_path / "research.db"
            db_dst = checkpoint_dir / "research.db"
            if db_src.exists():
                shutil.copy2(str(db_src), str(db_dst))
            reports_src = self.base_path / "reports"
            if reports_src.exists():
                reports_dst = checkpoint_dir / "reports"
                if reports_dst.exists():
                    shutil.rmtree(reports_dst)
                shutil.copytree(str(reports_src), str(reports_dst))

        await asyncio.to_thread(_copy)

        meta_path = checkpoint_dir / "metadata.json"
        meta_path.write_text(
            json.dumps({
                "name": name,
                "timestamp": datetime.now().isoformat(),
                "topic": self._current_topic,
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(checkpoint_dir)

    async def restore_checkpoint(self, name: str) -> bool:
        checkpoint_dir = self.base_path / "checkpoints" / name
        cp_db = checkpoint_dir / "research.db"
        if not cp_db.exists():
            return False

        def _restore():
            self._conn.close()
            shutil.copy2(str(cp_db), str(self.base_path / "research.db"))
            self._conn = sqlite3.connect(str(self.base_path / "research.db"), check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")

        await asyncio.to_thread(_restore)

        # 恢复 topic 上下文
        meta_path = checkpoint_dir / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("topic"):
                self._current_topic = meta["topic"]

        return True

    def get_workspace_info(self) -> Dict[str, Any]:
        rt = self.research_topic
        lit_count = self._conn.execute(
            "SELECT COUNT(*) FROM literature"
        ).fetchone()[0]
        clust_count = self._conn.execute(
            "SELECT COUNT(*) FROM clusters WHERE research_topic = ?", (rt,)
        ).fetchone()[0]
        summ_count = self._conn.execute(
            "SELECT COUNT(*) FROM summaries WHERE research_topic = ?", (rt,)
        ).fetchone()[0]
        emb_count = self._conn.execute(
            "SELECT COUNT(*) FROM embeddings"
        ).fetchone()[0]
        return {
            "topic": self._current_topic,
            "research_topic": rt,
            "literature_count": lit_count,
            "cluster_count": clust_count,
            "summary_count": summ_count,
            "embedding_count": emb_count,
            "base_path": str(self.base_path),
        }
