"""
Cluster Agent - 聚类分析Agent
负责将筛选后的文献按语义相似度自动聚类，发现主题结构
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult
from ..core.workspace import LiteratureRecord, ClusterResult, SharedWorkspace
from ..utils.embedding import get_embeddings, compute_similarity_matrix
from ..utils.llm import get_llm_client

logger = logging.getLogger(__name__)


class ClusterAgent(BaseAgent):
    """
    聚类分析Agent

    职责：
    ① 降维（t-SNE/UMAP）
    ② HDBSCAN聚类
    ③ 簇标签生成
    ④ 噪声点处理
    ⑤ 聚类质量评估
    """

    # 簇标签生成Prompt
    CLUSTER_LABELING_PROMPT = """你是一个文献分类专家。给定一个论文簇中的所有标题和关键摘要片段，
请为该簇生成：

1. 一个简洁的簇标签（不超过10个词）
2. 该簇的核心研究主题描述（2-3句话）
3. 该簇中的代表性论文（最多3篇）
4. 子主题列表

簇内论文：
{papers_info}

输出JSON格式：
{{
  "cluster_label": "簇标签",
  "core_theme": "核心主题描述",
  "representative_papers": [
    {{"title": "...", "reason": "代表性原因"}}
  ],
  "sub_themes": ["子主题1", "子主题2"]
}}"""

    def __init__(
        self,
        workspace: SharedWorkspace,
        llm_client: Optional[ChatOpenAI] = None,
        config: Optional[AgentConfig] = None,
    ):
        """初始化Cluster Agent"""
        config = config or AgentConfig(
            name="ClusterAgent",
            description="Clusters papers by semantic similarity",
            model="glm-4-flash",
            temperature=0.5,
        )
        super().__init__(config, workspace, llm_client)

        # 聚类参数（来自论文Table 7）
        self.default_params = {
            "dimensionality_reduction": {
                "method": "tsne",
                "n_components": 2,
                "perplexity": 30,
            },
            "clustering": {
                "method": "hdbscan",
                "min_cluster_size": 5,
                "min_samples": 3,
            },
        }

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return True

    async def execute(self, **kwargs) -> AgentResult:
        """
        执行聚类分析

        Args:
            method: 聚类方法（hdbscan, kmeans, spectral）
            min_cluster_size: 最小簇大小
            n_clusters: 指定簇数量（仅用于kmeans）

        Returns:
            AgentResult
        """
        method = kwargs.get("method", "hdbscan")
        min_cluster_size = kwargs.get("min_cluster_size", 5)

        try:
            # 获取相关文献
            papers = await self.workspace.get_literature()
            if not papers:
                return self._create_result(
                    success=False,
                    errors=["No papers in workspace"]
                )

            self.log_progress(f"Clustering {len(papers)} papers...")

            # ① 获取embeddings
            self.log_progress("Loading embeddings...")
            embeddings_dict = await self.workspace.get_all_embeddings()
            self.log_progress(f"Loaded {len(embeddings_dict)} embeddings from workspace")

            if not embeddings_dict:
                # 如果没有embeddings，生成它们
                self.log_progress("Generating embeddings...")

                # 临时限制：最多处理 500 篇论文（测试用）
                papers_limited = papers[:500] if len(papers) > 500 else papers
                self.log_progress(f"Limited to {len(papers_limited)} papers for testing")

                texts = [f"{p.title}\n{p.abstract or ''}" for p in papers_limited]
                self.log_progress(f"Prepared {len(texts)} texts for embedding")

                self.log_progress("Calling get_embeddings...")
                import time
                start_time = time.time()
                embeddings = await get_embeddings(texts)
                elapsed = time.time() - start_time
                self.log_progress(f"get_embeddings returned in {elapsed:.2f}s: {len(embeddings)} embeddings")

                self.log_progress("Creating embeddings dictionary...")
                embeddings_dict = {p.id: emb for p, emb in zip(papers_limited, embeddings)}
                self.log_progress(f"Dictionary created with {len(embeddings_dict)} entries")

            # 过滤有embedding的论文
            self.log_progress("Filtering papers with embeddings...")
            valid_papers = [p for p in papers if p.id in embeddings_dict]
            valid_embeddings = [embeddings_dict[p.id] for p in valid_papers]
            self.log_progress(f"Filtered to {len(valid_papers)} valid papers")

            if len(valid_papers) < min_cluster_size:
                return self._create_result(
                    success=False,
                    errors=[f"Not enough papers for clustering: {len(valid_papers)} < {min_cluster_size}"]
                )

            # ② 降维
            self.log_progress("Performing dimensionality reduction...")
            reduced_embeddings = await self._dimensionality_reduction(valid_embeddings)

            # ③ 聚类
            self.log_progress(f"Performing {method.upper()} clustering...")
            cluster_labels = await self._perform_clustering(
                reduced_embeddings,
                method=method,
                min_cluster_size=min_cluster_size,
            )

            # ④ 生成簇信息
            self.log_progress("Generating cluster information...")
            clusters = await self._generate_clusters(
                valid_papers,
                cluster_labels,
                reduced_embeddings,
            )

            # ⑤ 生成簇标签
            self.log_progress("Generating cluster labels...")
            labeled_clusters = await self._label_clusters(clusters)

            # 保存聚类结果
            await self.workspace.save_clusters(labeled_clusters)

            # ⑥ 聚类质量评估
            metrics = await self._evaluate_clustering(
                valid_embeddings,
                cluster_labels,
                reduced_embeddings,
            )

            self.log_progress(
                f"Clustering complete: {len(labeled_clusters)} clusters found, "
                f"{metrics.get('noise', 0)} noise points"
            )

            # 保存可视化数据
            await self._save_visualization_data(
                valid_papers,
                reduced_embeddings,
                cluster_labels,
            )

            return self._create_result(
                success=True,
                data={
                    "cluster_count": len(labeled_clusters),
                    "clusters": [c.to_dict() for c in labeled_clusters],
                },
                metrics=metrics,
            )

        except KeyboardInterrupt:
            raise  # 重新抛出，让 BaseAgent.run() 处理
        except Exception as e:
            error_msg = f"Clustering execution failed: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

    async def _dimensionality_reduction(
        self,
        embeddings: List[List[float]],
    ) -> np.ndarray:
        """
        降维

        Args:
            embeddings: 原始embeddings

        Returns:
            降维后的数组
        """
        try:
            from sklearn.manifold import TSNE
            from sklearn.decomposition import PCA

            arr = np.array(embeddings)
            n_samples = len(arr)

            self.log_progress(f"Starting dimensionality reduction for {n_samples} papers...")

            # 如果维度太高，先用PCA降到50维
            if arr.shape[1] > 50:
                self.log_progress(f"Reducing from {arr.shape[1]} to 50 dimensions using PCA...")
                pca = PCA(n_components=50, random_state=42)
                arr = pca.fit_transform(arr)
                self.log_progress(f"PCA reduction complete")

            # 根据数据量选择降维方法
            if n_samples > 1000:
                # 大数据量：直接用PCA降到2维（快速）
                self.log_progress(f"Large dataset ({n_samples} samples), using PCA for final reduction...")
                pca_final = PCA(n_components=2, random_state=42)
                reduced = pca_final.fit_transform(arr)
            else:
                # 小数据量：用t-SNE
                self.log_progress(f"Using t-SNE for final reduction ({n_samples} samples)...")
                tsne = TSNE(
                    n_components=2,
                    perplexity=min(30, max(5, n_samples // 10)),
                    random_state=42,
                    max_iter=500,  # 减少迭代次数加快速度
                )
                reduced = tsne.fit_transform(arr)

            self.log_progress(f"Dimensionality reduction complete: {reduced.shape[1]} dimensions")
            return reduced

        except ImportError:
            self.log_progress("scikit-learn not available, using fallback", "warning")
            # 简单fallback：选择前两个维度
            return np.array(embeddings)[:, :2]

    async def _perform_clustering(
        self,
        embeddings: np.ndarray,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
    ) -> np.ndarray:
        """
        执行聚类

        Args:
            embeddings: 降维后的embeddings
            method: 聚类方法
            min_cluster_size: 最小簇大小

        Returns:
            聚类标签数组
        """
        try:
            import hdbscan

            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            labels = clusterer.fit_predict(embeddings)

            self.log_progress(f"HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
            return labels

        except ImportError:
            self.log_progress("HDBSCAN not available, using KMeans fallback", "warning")
            # KMeans fallback
            try:
                from sklearn.cluster import KMeans

                # 估计簇数
                n_clusters = max(3, len(embeddings) // min_cluster_size)

                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(embeddings)

                self.log_progress(f"KMeans found {n_clusters} clusters")
                return labels

            except ImportError:
                # 最简单的fallback：基于距离的简单聚类
                self.log_progress("No clustering library available, using simple distance-based clustering", "warning")
                return self._simple_clustering(embeddings, min_cluster_size)

    def _simple_clustering(
        self,
        embeddings: np.ndarray,
        min_cluster_size: int,
    ) -> np.ndarray:
        """
        简单的距离聚类（fallback）

        Args:
            embeddings: embeddings数组
            min_cluster_size: 最小簇大小

        Returns:
            聚类标签
        """
        n = len(embeddings)
        labels = np.full(n, -1)  # -1表示噪声
        current_label = 0

        # 计算距离矩阵
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(embeddings))

        # 简单的层次聚类
        visited = set()
        threshold = np.percentile(distances, 20)  # 使用20%分位数作为阈值

        for i in range(n):
            if i in visited:
                continue

            # 找到所有在阈值内的点
            neighbors = np.where(distances[i] < threshold)[0]

            if len(neighbors) >= min_cluster_size:
                # 形成一个簇
                for neighbor in neighbors:
                    labels[neighbor] = current_label
                    visited.add(neighbor)
                current_label += 1

        return labels

    async def _generate_clusters(
        self,
        papers: List[LiteratureRecord],
        labels: np.ndarray,
        reduced_embeddings: np.ndarray,
    ) -> List[ClusterResult]:
        """
        生成簇信息

        Args:
            papers: 论文列表
            labels: 聚类标签
            reduced_embeddings: 降维后的embeddings

        Returns:
            簇列表
        """
        clusters = {}
        noise_count = 0

        for paper, label, emb in zip(papers, labels, reduced_embeddings):
            if label == -1:
                noise_count += 1
                continue

            if label not in clusters:
                clusters[label] = {
                    "paper_ids": [],
                    "papers": [],
                    "embeddings": [],
                }

            clusters[label]["paper_ids"].append(paper.id)
            clusters[label]["papers"].append(paper)
            clusters[label]["embeddings"].append(emb)

        results = []
        for label, data in clusters.items():
            results.append(ClusterResult(
                cluster_id=label,
                label="",  # 将在labeling步骤填充
                description="",  # 将在labeling步骤填充
                paper_ids=data["paper_ids"],
                representative_papers=[],
                sub_themes=[],
                size=len(data["paper_ids"]),
            ))

        self.log_progress(f"Generated {len(results)} clusters, {noise_count} noise points")
        return results

    async def _label_clusters(
        self,
        clusters: List[ClusterResult],
    ) -> List[ClusterResult]:
        """
        为簇生成标签

        Args:
            clusters: 簇列表

        Returns:
            带标签的簇列表
        """
        for cluster in clusters:
            try:
                # 获取簇内论文
                papers = await self.workspace.get_literature(paper_ids=cluster.paper_ids)

                # 准备论文信息
                papers_info = "\n".join([
                    f"- {p.title} (Year: {p.year})"
                    for p in papers[:20]  # 限制数量避免token超限
                ])

                # 调用LLM生成标签
                messages = [
                    SystemMessage(content="You are an expert in research paper classification."),
                    HumanMessage(
                        content=self.CLUSTER_LABELING_PROMPT.format(papers_info=papers_info)
                    ),
                ]

                response = await self._call_llm(messages, response_format="json")
                result = json.loads(response)

                # 更新簇信息
                cluster.label = result.get("cluster_label", f"Cluster {cluster.cluster_id}")
                cluster.description = result.get("core_theme", "")
                cluster.representative_papers = result.get("representative_papers", [])
                cluster.sub_themes = result.get("sub_themes", [])

            except Exception as e:
                self.log_progress(f"Failed to label cluster {cluster.cluster_id}: {e}", "warning")
                cluster.label = f"Cluster {cluster.cluster_id}"
                cluster.description = f"A cluster of {cluster.size} papers on related topics"

        return clusters

    async def _evaluate_clustering(
        self,
        embeddings: List[List[float]],
        labels: np.ndarray,
        reduced_embeddings: np.ndarray,
    ) -> Dict[str, float]:
        """
        评估聚类质量

        Args:
            embeddings: 原始embeddings
            labels: 聚类标签
            reduced_embeddings: 降维后的embeddings

        Returns:
            质量指标字典
        """
        metrics = {}

        try:
            from sklearn.metrics import silhouette_score, davies_bouldin_score

            # 轮廓系数
            if len(set(labels)) > 1 and -1 not in labels:
                silhouette = silhouette_score(reduced_embeddings, labels)
                metrics["silhouette_score"] = float(silhouette)
            else:
                metrics["silhouette_score"] = 0.0

            # Davies-Bouldin指数
            if len(set(labels)) > 1:
                db_score = davies_bouldin_score(reduced_embeddings, labels)
                metrics["davies_bouldin_score"] = float(db_score)

            # 簇数量
            metrics["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)

            # 噪声点数量
            metrics["noise"] = int(np.sum(labels == -1))

        except ImportError:
            self.log_progress("scikit-learn not available for metrics", "warning")
            metrics["n_clusters"] = len(set(labels)) - (1 if -1 in labels else 0)
            metrics["noise"] = int(np.sum(labels == -1))

        return metrics

    async def _save_visualization_data(
        self,
        papers: List[LiteratureRecord],
        reduced_embeddings: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """
        保存可视化数据

        Args:
            papers: 论文列表
            reduced_embeddings: 降维后的坐标
            labels: 聚类标签
        """
        viz_data = {
            "papers": [
                {
                    "id": p.id,
                    "title": p.title,
                    "x": float(coord[0]),
                    "y": float(coord[1]),
                    "cluster": int(label),
                }
                for p, coord, label in zip(papers, reduced_embeddings, labels)
            ]
        }

        await self._save_to_workspace("cluster_visualization", viz_data, stage="cluster")

    async def get_cluster_summary(self, cluster_id: int) -> Optional[Dict[str, Any]]:
        """
        获取簇摘要

        Args:
            cluster_id: 簇ID

        Returns:
            簇摘要信息
        """
        cluster = await self.workspace.get_cluster(cluster_id)
        if not cluster:
            return None

        papers = await self.workspace.get_cluster_papers(cluster_id)

        return {
            "cluster": cluster.to_dict(),
            "papers": [p.to_dict() for p in papers],
            "paper_count": len(papers),
        }
