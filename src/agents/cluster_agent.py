"""
Cluster Agent - 聚类分析Agent
负责将筛选后的文献按语义相似度自动聚类，发现主题结构

GPU加速支持：优先使用 PyTorch CUDA，不可用时回退到 sklearn/hdbscan
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult, get_agent_model
from ..core.workspace import LiteratureRecord, ClusterResult, SharedWorkspace
from ..utils.embedding import get_embeddings, compute_similarity_matrix
from ..utils.llm import get_llm_client

logger = logging.getLogger(__name__)

# GPU 后端检测（PyTorch CUDA）
_GPU_BACKEND = "cpu"
_torch_device = None
try:
    import torch
    if torch.cuda.is_available():
        _GPU_BACKEND = "torch-cuda"
        _torch_device = torch.device("cuda")
        logger.info(f"PyTorch CUDA GPU backend available: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("CUDA not available, using CPU backend")
except ImportError:
    logger.info("PyTorch not available, using CPU backend")


def _gpu_pca(data: np.ndarray, n_components: int) -> np.ndarray:
    """使用 PyTorch GPU 进行 PCA 降维"""
    import torch

    t = torch.tensor(data, dtype=torch.float32, device=_torch_device)
    # 中心化
    mean = t.mean(dim=0, keepdim=True)
    t = t - mean
    # SVD 分解（GPU 加速）
    U, S, Vh = torch.linalg.svd(t, full_matrices=False)
    # 取前 n_components 个主成分
    result = U[:, :n_components] * S[:n_components]
    return result.cpu().numpy()


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
5. 该簇论文共享的核心公式或数学框架（如果可以从摘要中识别）
6. 该簇方法论在原理层面的共同点（底层物理/数学原理，而非表面方法名）

簇内论文：
{papers_info}

输出JSON格式：
{{
  "cluster_label": "簇标签",
  "core_theme": "核心主题描述",
  "representative_papers": [
    {{"title": "...", "reason": "代表性原因"}}
  ],
  "sub_themes": ["子主题1", "子主题2"],
  "shared_formulas": "该簇涉及的核心公式或数学框架描述",
  "underlying_principles": "该簇论文共享的底层物理/数学原理"
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
            model=get_agent_model("cluster"),
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
                "min_cluster_size": 6,
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
        research_topic = kwargs.get("research_topic", "")

        try:
            # 只获取筛选后的相关文献（relevance_score 不为 None）
            all_papers = await self.workspace.get_literature()
            papers = [p for p in all_papers if p.relevance_score is not None]
            if not papers:
                return self._create_result(
                    success=False,
                    errors=["No relevant papers in workspace"]
                )

            # 自适应 min_cluster_size：确保能形成簇
            n_papers = len(papers)
            if min_cluster_size > n_papers // 3:
                min_cluster_size = max(2, n_papers // 10)
                self.log_progress(
                    f"Adjusted min_cluster_size to {min_cluster_size} "
                    f"(only {n_papers} papers available)"
                )

            self.log_progress(f"Clustering {len(papers)} papers...")

            # ① 获取embeddings
            self.log_progress("Loading embeddings...")
            embeddings_dict = await self.workspace.get_all_embeddings()
            # 只保留当前论文的embedding
            embeddings_dict = {pid: emb for pid, emb in embeddings_dict.items()
                               if pid in {p.id for p in papers}}
            self.log_progress(f"Loaded {len(embeddings_dict)} cached embeddings for current papers")

            missing_papers = [p for p in papers if p.id not in embeddings_dict]
            if missing_papers:
                # 生成缺失的embeddings
                self.log_progress(f"Generating embeddings for {len(missing_papers)} papers...")

                texts = [f"{p.title}\n{p.abstract or ''}" for p in missing_papers]
                self.log_progress(f"Prepared {len(texts)} texts for embedding")

                self.log_progress("Calling get_embeddings...")
                import time
                start_time = time.time()
                embeddings = await get_embeddings(texts)
                elapsed = time.time() - start_time
                self.log_progress(f"get_embeddings returned in {elapsed:.2f}s: {len(embeddings)} embeddings")

                self.log_progress("Creating embeddings dictionary...")
                new_embeddings = {p.id: emb for p, emb in zip(missing_papers, embeddings)}
                embeddings_dict.update(new_embeddings)
                self.log_progress(f"Total embeddings: {len(embeddings_dict)}")

            # 不限制论文数量，全部参与聚类
            self.log_progress(f"Clustering all {len(embeddings_dict)} papers")

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

            # ⑤.5 领域一致性过滤：移除与主题明显不相关的簇
            labeled_clusters = await self._filter_irrelevant_clusters(
                labeled_clusters, research_topic
            )

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

            # 保存可视化数据（将被过滤掉的簇标记为噪声 -1）
            filtered_ids = {c.cluster_id for c in labeled_clusters}
            viz_labels = np.where(np.isin(cluster_labels, list(filtered_ids)), cluster_labels, -1)
            await self._save_visualization_data(
                valid_papers,
                reduced_embeddings,
                viz_labels,
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
        降维（PCA 优先 GPU，t-SNE 使用 CPU）

        Args:
            embeddings: 原始embeddings

        Returns:
            降维后的数组
        """
        arr = np.array(embeddings, dtype=np.float32)
        n_samples = len(arr)

        self.log_progress(f"Starting dimensionality reduction for {n_samples} papers ({_GPU_BACKEND})...")

        # 论文太少时直接用 PCA，跳过 t-SNE（样本不足会导致 t-SNE 失败）
        if n_samples < 10:
            self.log_progress(f"Too few samples ({n_samples}) for t-SNE, using PCA directly...")
            if _GPU_BACKEND == "torch-cuda":
                return _gpu_pca(arr, 2)
            else:
                from sklearn.decomposition import PCA
                return PCA(n_components=2, random_state=42).fit_transform(arr)

        # 第一步：高维 → 50维（PCA，优先 GPU）
        if arr.shape[1] > 50:
            if _GPU_BACKEND == "torch-cuda":
                self.log_progress(f"PCA {arr.shape[1]}→50 (GPU/PyTorch)...")
                arr = _gpu_pca(arr, 50)
            else:
                from sklearn.decomposition import PCA
                self.log_progress(f"PCA {arr.shape[1]}→50 (CPU/sklearn)...")
                arr = PCA(n_components=50, random_state=42).fit_transform(arr)

        # 第二步：50维 → 2维（统一使用 t-SNE）
        from sklearn.manifold import TSNE
        self.log_progress(f"t-SNE 50→2 ({n_samples} samples)...")
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_samples // 10)),
            random_state=42,
            max_iter=500,
        )
        reduced = tsne.fit_transform(arr)

        reduced = np.array(reduced, dtype=np.float64)
        self.log_progress(f"Dimensionality reduction complete: {reduced.shape[1]} dimensions ({_GPU_BACKEND})")
        return reduced

    async def _perform_clustering(
        self,
        embeddings: np.ndarray,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
    ) -> np.ndarray:
        """
        执行聚类（hdbscan CPU，HDBSCAN 无 GPU 替代方案）

        Args:
            embeddings: 降维后的embeddings
            method: 聚类方法
            min_cluster_size: 最小簇大小

        Returns:
            聚类标签数组
        """
        data = np.array(embeddings, dtype=np.float32)

        try:
            import hdbscan

            self.log_progress("Using hdbscan (CPU)...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=3,
                metric='euclidean',
                cluster_selection_method='eom',
            )
            labels = clusterer.fit_predict(data)

            self.log_progress(f"HDBSCAN found {len(set(labels)) - (1 if -1 in labels else 0)} clusters")
            return labels

        except ImportError:
            self.log_progress("HDBSCAN not available, using KMeans fallback", "warning")
            try:
                from sklearn.cluster import KMeans

                n_clusters = max(3, len(data) // min_cluster_size)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                labels = kmeans.fit_predict(data)

                self.log_progress(f"KMeans found {n_clusters} clusters")
                return labels

            except ImportError:
                self.log_progress("No clustering library available, using simple distance-based clustering", "warning")
                return self._simple_clustering(data, min_cluster_size)

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
                cluster_id=int(label),
                label="",  # 将在labeling步骤填充
                description="",  # 将在labeling步骤填充
                paper_ids=data["paper_ids"],
                representative_papers=[],
                sub_themes=[],
                size=len(data["paper_ids"]),
            ))

        self.log_progress(f"Generated {len(results)} clusters, {noise_count} noise points")
        return results

    async def _filter_irrelevant_clusters(
        self,
        clusters: List[ClusterResult],
        research_topic: str,
    ) -> List[ClusterResult]:
        """
        过滤与主题明显不相关的簇。

        对每个簇，检查其代表性论文标题是否与研究主题相关。
        如果簇内超过半数论文的标题中不包含主题核心词，则移除该簇。
        """
        if not research_topic or not clusters:
            return clusters

        # 提取主题核心词（去除停用词）
        stop_words = {"of", "the", "a", "an", "in", "for", "and", "or", "to",
                       "with", "on", "by", "from", "at", "is", "are", "was", "were"}
        topic_words = set(w.lower() for w in research_topic.split()
                          if w.lower() not in stop_words and len(w) > 2)

        filtered = []
        for cluster in clusters:
            papers = await self.workspace.get_cluster_papers(cluster.cluster_id)
            if not papers:
                filtered.append(cluster)
                continue

            relevant_count = 0
            for paper in papers:
                title_words = set(w.lower() for w in paper.title.split()
                                  if len(w) > 2)
                # 检查标题是否与主题有交集
                if topic_words & title_words:
                    relevant_count += 1

            ratio = relevant_count / len(papers)
            if ratio >= 0.3:
                filtered.append(cluster)
            else:
                self.log_progress(
                    f"Filtered out cluster {cluster.cluster_id} '{cluster.label}': "
                    f"only {relevant_count}/{len(papers)} papers match topic keywords",
                    "warning",
                )

        if len(filtered) < len(clusters):
            self.log_progress(
                f"Filtered {len(clusters) - len(filtered)} irrelevant clusters, "
                f"keeping {len(filtered)}"
            )

        return filtered

    async def _label_clusters(
        self,
        clusters: List[ClusterResult],
    ) -> List[ClusterResult]:
        """
        为簇生成标签（并行）

        Args:
            clusters: 簇列表

        Returns:
            带标签的簇列表
        """
        async def _label_one(cluster: ClusterResult) -> None:
            try:
                papers = await self.workspace.get_literature(paper_ids=cluster.paper_ids)
                papers_info = "\n".join([
                    f"- {p.title} (Year: {p.year})"
                    for p in papers[:20]
                ])
                messages = [
                    SystemMessage(content="You are an expert in research paper classification."),
                    HumanMessage(
                        content=self.CLUSTER_LABELING_PROMPT.format(papers_info=papers_info)
                    ),
                ]
                response = await self._call_llm(messages, response_format="json")
                result = json.loads(response)
                cluster.label = result.get("cluster_label", f"Cluster {cluster.cluster_id}")
                cluster.description = result.get("core_theme", "")
                cluster.representative_papers = result.get("representative_papers", [])
                cluster.sub_themes = result.get("sub_themes", [])
            except Exception as e:
                self.log_progress(f"Failed to label cluster {cluster.cluster_id}: {e}", "warning")
                cluster.label = f"Cluster {cluster.cluster_id}"
                cluster.description = f"A cluster of {cluster.size} papers on related topics"

        self.log_progress(f"Labeling {len(clusters)} clusters in parallel...")
        await asyncio.gather(*[_label_one(c) for c in clusters])
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

            # 轮廓系数（过滤噪声点后计算）
            non_noise_mask = labels != -1
            non_noise_labels = labels[non_noise_mask]
            if len(set(non_noise_labels)) > 1:
                silhouette = silhouette_score(
                    reduced_embeddings[non_noise_mask], non_noise_labels
                )
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
