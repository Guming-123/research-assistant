"""
Embedding utilities
向量嵌入和相似度计算
"""

import asyncio
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np

from .llm import get_embedding_client

logger = logging.getLogger(__name__)


async def get_embeddings(
    texts: List[str],
    model: str = "text-embedding-3-small",
    batch_size: int = 100,
) -> List[List[float]]:
    """
    获取文本的embeddings

    Args:
        texts: 文本列表
        model: 模型名称
        batch_size: 批处理大小

    Returns:
        embedding向量列表
    """
    embedding_client = get_embedding_client(model)

    all_embeddings = []

    # 分批处理
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            embeddings = await embedding_client.aembed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.debug(f"Generated embeddings for batch {i//batch_size + 1}")
        except Exception as e:
            logger.error(f"Failed to generate embeddings for batch: {e}")
            # 返回零向量作为fallback
            zero_embedding = [0.0] * 1536  # text-embedding-3-small的维度
            all_embeddings.extend([zero_embedding] * len(batch))

    return all_embeddings


async def get_embedding(text: str, model: str = "text-embedding-3-small") -> List[float]:
    """
    获取单个文本的embedding

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        embedding向量
    """
    embedding_client = get_embedding_client(model)
    try:
        return await embedding_client.aembed_query(text)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        return [0.0] * 1536


def compute_similarity(
    vec1: List[float],
    vec2: List[float],
    method: str = "cosine",
) -> float:
    """
    计算两个向量的相似度

    Args:
        vec1: 向量1
        vec2: 向量2
        method: 相似度计算方法（cosine, euclidean, dot）

    Returns:
        相似度值
    """
    arr1 = np.array(vec1)
    arr2 = np.array(vec2)

    if method == "cosine":
        # 余弦相似度
        dot_product = np.dot(arr1, arr2)
        norm1 = np.linalg.norm(arr1)
        norm2 = np.linalg.norm(arr2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    elif method == "euclidean":
        # 欧氏距离（转换为相似度）
        distance = np.linalg.norm(arr1 - arr2)
        return 1 / (1 + distance)

    elif method == "dot":
        # 点积
        return np.dot(arr1, arr2)

    else:
        raise ValueError(f"Unknown similarity method: {method}")


def compute_similarity_matrix(
    embeddings: List[List[float]],
    method: str = "cosine",
) -> np.ndarray:
    """
    计算相似度矩阵

    Args:
        embeddings: embedding向量列表
        method: 相似度计算方法

    Returns:
        相似度矩阵
    """
    n = len(embeddings)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            sim = compute_similarity(embeddings[i], embeddings[j], method)
            matrix[i][j] = sim
            matrix[j][i] = sim

    return matrix


def find_most_similar(
    query_embedding: List[float],
    corpus_embeddings: List[List[float]],
    top_k: int = 5,
    method: str = "cosine",
) -> List[Tuple[int, float]]:
    """
    找到最相似的向量

    Args:
        query_embedding: 查询向量
        corpus_embeddings: 语料库向量列表
        top_k: 返回前k个结果
        method: 相似度计算方法

    Returns:
        [(索引, 相似度), ...] 列表
    """
    similarities = []
    for i, emb in enumerate(corpus_embeddings):
        sim = compute_similarity(query_embedding, emb, method)
        similarities.append((i, sim))

    # 按相似度排序
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]


def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """
    L2标准化embeddings

    Args:
        embeddings: embedding向量列表

    Returns:
        标准化后的向量列表
    """
    normalized = []
    for emb in embeddings:
        arr = np.array(emb)
        norm = np.linalg.norm(arr)
        if norm > 0:
            normalized.append((arr / norm).tolist())
        else:
            normalized.append(emb)
    return normalized


class FAISSIndex:
    """
    简化的FAISS索引封装

    使用numpy实现基础的向量检索功能
    （当FAISS不可用时的fallback方案）
    """

    def __init__(self, embeddings: List[List[float]], ids: Optional[List[str]] = None):
        """
        初始化索引

        Args:
            embeddings: embedding向量列表
            ids: 对应的ID列表
        """
        self.embeddings = np.array(embeddings)
        self.ids = ids or [str(i) for i in range(len(embeddings))]

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        搜索最相似的向量

        Args:
            query_embedding: 查询向量
            top_k: 返回前k个结果

        Returns:
            [(ID, 相似度), ...] 列表
        """
        query = np.array(query_embedding)

        # 计算与所有向量的余弦相似度
        similarities = []
        for i, emb in enumerate(self.embeddings):
            sim = compute_similarity(query, emb.tolist(), "cosine")
            similarities.append((self.ids[i], sim))

        # 排序并返回top-k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]


async def build_faiss_index(
    embeddings: List[List[float]],
    ids: Optional[List[str]] = None,
) -> FAISSIndex:
    """
    构建FAISS索引

    Args:
        embeddings: embedding向量列表
        ids: 对应的ID列表

    Returns:
        FAISSIndex实例
    """
    return FAISSIndex(embeddings, ids)
