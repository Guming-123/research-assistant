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

        # 簇标签生成使用 glm-4-flash，不受 glm-5 的 Coding 端点并发上限约束，
        # 与 ScreenAgent 一致恢复并发 15，加速并行打标。
        self.llm_semaphore = asyncio.Semaphore(15)

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
            # 只获取当前搜索中通过筛选的相关文献
            # 优先使用 current_search_paper_ids 限定范围，避免旧主题论文混入
            search_ids = await self.workspace.get_metadata_item("current_search_paper_ids")
            if search_ids and isinstance(search_ids, list):
                search_papers = await self.workspace.get_literature(paper_ids=search_ids)
                self.log_progress(
                    f"Scoped to {len(search_ids)} papers from current search"
                )
            else:
                # 回退：无搜索记录时加载全部，但先重置所有分数防止旧数据污染
                self.log_progress(
                    "No search session found, loading all papers with score reset",
                    "warning",
                )
                await self.workspace.reset_all_relevance_scores()
                search_papers = await self.workspace.get_literature()
            papers = [p for p in search_papers if p.relevance_score is not None]
            if not papers:
                return self._create_result(
                    success=False,
                    errors=["No relevant papers in workspace"]
                )

            # 自适应 min_cluster_size / min_samples：高维空间里样本少时 HDBSCAN
            # 容易把所有点判为噪声（出现 0 簇）。论文少时必须调小这两个参数。
            n_papers = len(papers)
            if n_papers < 200:
                min_cluster_size = max(3, min(min_cluster_size, n_papers // 15))
            min_samples = max(2, min(3, min_cluster_size - 1))
            self.log_progress(
                f"Adaptive clustering params for {n_papers} papers: "
                f"min_cluster_size={min_cluster_size}, min_samples={min_samples}"
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

            # ② 降维：聚类空间(高维) 与 可视化空间(2D) 分离
            #    关键改动：HDBSCAN 在高维(PCA→50)空间聚类，2D 仅用于散点图，
            #    避免在 t-SNE 2D 投影上聚类导致退化为少数超大簇。
            self.log_progress("Preparing clustering space (high-dim PCA)...")
            cluster_space = await self._reduce_for_clustering(valid_embeddings)
            self.log_progress("Preparing visualization space (2D)...")
            viz_embeddings = await self._dimensionality_reduction(valid_embeddings)

            # ③ 聚类（在高维空间进行，而非 2D）
            self.log_progress(
                f"Performing {method.upper()} clustering on {cluster_space.shape[1]}-dim space..."
            )
            cluster_labels = await self._perform_clustering(
                cluster_space,
                method=method,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            )

            # ④ 生成簇信息（reduced 坐标仅记录，传入 2D 可视化坐标）
            self.log_progress("Generating cluster information...")
            clusters = await self._generate_clusters(
                valid_papers,
                cluster_labels,
                viz_embeddings,
            )

            # ⑤ 生成簇标签
            self.log_progress("Generating cluster labels...")
            labeled_clusters = await self._label_clusters(clusters)

            # ⑤.5 领域一致性过滤：移除与主题明显不相关的簇
            labeled_clusters = await self._filter_irrelevant_clusters(
                labeled_clusters, research_topic, embeddings_dict
            )

            # 保存聚类结果
            await self.workspace.save_clusters(labeled_clusters)

            # ⑥ 聚类质量评估（轮廓系数等在「聚类空间」高维上计算更有意义）
            metrics = await self._evaluate_clustering(
                cluster_space,
                cluster_labels,
                cluster_space,
            )

            self.log_progress(
                f"Clustering complete: {len(labeled_clusters)} clusters found, "
                f"{metrics.get('noise', 0)} noise points"
            )

            # 保存可视化数据（将被过滤掉的簇标记为噪声 -1）；坐标用 2D 可视化空间
            filtered_ids = {c.cluster_id for c in labeled_clusters}
            viz_labels = np.where(np.isin(cluster_labels, list(filtered_ids)), cluster_labels, -1)
            await self._save_visualization_data(
                valid_papers,
                viz_embeddings,
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

    def _pca_reduce(self, arr: np.ndarray, n_components: int) -> np.ndarray:
        """PCA 降维（GPU 优先，CPU 回退）"""
        if _GPU_BACKEND == "torch-cuda":
            return _gpu_pca(arr, n_components)
        from sklearn.decomposition import PCA
        return PCA(n_components=n_components, random_state=42).fit_transform(arr)

    def _tsne_to_2d(self, arr: np.ndarray) -> np.ndarray:
        """t-SNE → 2D（仅用于可视化，不用于聚类）"""
        from sklearn.manifold import TSNE
        n_samples = len(arr)
        tsne = TSNE(
            n_components=2,
            perplexity=min(30, max(5, n_samples // 10)),
            random_state=42,
            max_iter=500,
        )
        return tsne.fit_transform(arr)

    async def _reduce_for_clustering(
        self,
        embeddings: List[List[float]],
    ) -> np.ndarray:
        """
        生成「聚类空间」：高维 → 自适应目标维度的 PCA。

        聚类维度随样本量自适应（10–50 维）：样本少时降到更低维，
        缓解维度灾难导致 HDBSCAN 把所有点判为噪声（0 簇）。
        HDBSCAN 在此空间聚类，2D 仅用于可视化。
        """
        arr = np.array(embeddings, dtype=np.float32)
        n = len(arr)
        target = max(10, min(50, n // 5))  # 50篇→10维, 200篇→40维, 250+篇→50维
        if arr.shape[1] <= target:
            self.log_progress(f"Clustering space: keep {arr.shape[1]}-dim (<= {target})")
            return arr
        self.log_progress(f"PCA {arr.shape[1]}→{target} for clustering ({_GPU_BACKEND}, n={n})...")
        return self._pca_reduce(arr, target)

    async def _dimensionality_reduction(
        self,
        embeddings: List[List[float]],
    ) -> np.ndarray:
        """
        生成「可视化空间」：→ 2D（t-SNE，样本过少时 PCA）。

        仅用于散点图与可视化，不参与聚类决策。
        """
        arr = np.array(embeddings, dtype=np.float32)
        n_samples = len(arr)

        self.log_progress(f"Preparing 2D visualization for {n_samples} papers ({_GPU_BACKEND})...")

        # 论文太少时直接用 PCA，跳过 t-SNE（样本不足会导致 t-SNE 失败）
        if n_samples < 10:
            self.log_progress(f"Too few samples ({n_samples}) for t-SNE, using PCA directly...")
            return self._pca_reduce(arr, 2)

        # 第一步：高维 → 50维（PCA，优先 GPU）
        if arr.shape[1] > 50:
            arr = self._pca_reduce(arr, 50)

        # 第二步：50维 → 2维（t-SNE，仅可视化）
        reduced = self._tsne_to_2d(arr)
        reduced = np.array(reduced, dtype=np.float64)
        self.log_progress(f"2D visualization ready ({_GPU_BACKEND})")
        return reduced

    async def _perform_clustering(
        self,
        embeddings: np.ndarray,
        method: str = "hdbscan",
        min_cluster_size: int = 5,
        min_samples: int = 3,
    ) -> np.ndarray:
        """
        执行聚类（hdbscan CPU，HDBSCAN 无 GPU 替代方案）

        Args:
            embeddings: 降维后的embeddings
            method: 聚类方法
            min_cluster_size: 最小簇大小
            min_samples: HDBSCAN min_samples（核心点邻域样本数）

        Returns:
            聚类标签数组
        """
        data = np.array(embeddings, dtype=np.float32)

        try:
            import hdbscan

            self.log_progress("Using hdbscan (CPU)...")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
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
        embeddings_dict: Optional[Dict[str, List[float]]] = None,
    ) -> List[ClusterResult]:
        """
        过滤与主题明显不相关的簇。

        策略：计算每个簇内论文 embedding 与研究主题 embedding 的
        平均余弦相似度。低于阈值则移除该簇。回退到关键词匹配。
        """
        if not research_topic or not clusters:
            return clusters

        # 优先使用 embedding 语义过滤
        if embeddings_dict:
            try:
                from ..utils.embedding import get_embedding

                topic_embedding = await get_embedding(research_topic)
                topic_arr = np.array(topic_embedding, dtype=np.float32)
                t_norm = np.linalg.norm(topic_arr)
                if t_norm > 0:
                    topic_arr = topic_arr / t_norm

                min_cluster_similarity = 0.45  # 簇平均相似度下限
                filtered = []
                for cluster in clusters:
                    papers = await self.workspace.get_cluster_papers(cluster.cluster_id)
                    if not papers:
                        filtered.append(cluster)
                        continue

                    # 计算簇内每篇论文与主题的余弦相似度
                    sims = []
                    for p in papers:
                        emb = embeddings_dict.get(p.id)
                        if emb:
                            p_arr = np.array(emb, dtype=np.float32)
                            p_norm = np.linalg.norm(p_arr)
                            if p_norm > 0:
                                sims.append(float(np.dot(p_arr / p_norm, topic_arr)))

                    avg_sim = sum(sims) / len(sims) if sims else 0.0

                    if avg_sim >= min_cluster_similarity:
                        filtered.append(cluster)
                    else:
                        self.log_progress(
                            f"Filtered out cluster {cluster.cluster_id} '{cluster.label}': "
                            f"avg similarity to topic = {avg_sim:.3f} < {min_cluster_similarity}",
                            "warning",
                        )

                if len(filtered) < len(clusters):
                    self.log_progress(
                        f"Semantic filtering: removed {len(clusters) - len(filtered)} "
                        f"off-topic clusters, keeping {len(filtered)}"
                    )
                return filtered

            except Exception as e:
                self.log_progress(
                    f"Embedding-based filtering failed ({e}), falling back to keywords",
                    "warning",
                )

        # 回退：关键词匹配（原有逻辑）
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
                f"Keyword filtering: removed {len(clusters) - len(filtered)} clusters, "
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
