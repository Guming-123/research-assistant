"""
Embedding utilities
向量嵌入和相似度计算

支持：
1. 本地开源模型（sentence-transformers, FlagEmbedding等）
2. 远程API（GLM, OpenAI等）
"""

import asyncio
from typing import List, Dict, Optional, Tuple
import logging
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

# 全局变量缓存本地模型
_local_models = {}
_executor = ThreadPoolExecutor(max_workers=1)


def _get_local_model(model_name: str):
    """获取或加载本地模型"""
    global _local_models

    if model_name in _local_models:
        return _local_models[model_name]

    try:
        # 检测是否可用GPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        if model_name.startswith("bge-"):
            # 使用 FlagEmbedding (BGE系列)
            from FlagEmbedding import FlagModel
            model = FlagModel(
                f'BAAI/{model_name}',
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                device=device  # 使用GPU
            )
            _local_models[model_name] = model
            logger.info(f"Loaded BGE model: {model_name} on {device}")
            return model

        elif model_name.startswith("m3e-"):
            # 使用 M3E 模型
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer(f'moka-ai/{model_name}', device=device)
            _local_models[model_name] = model
            logger.info(f"Loaded M3E model: {model_name} on {device}")
            return model

        else:
            # 默认使用 sentence-transformers
            from sentence_transformers import SentenceTransformer

            # 模型名称映射
            model_map = {
                "local": "paraphrase-multilingual-MiniLM-L12-v2",  # 多语言
                "local-zh": "BAAI/bge-small-zh-v1.5",  # 中文优化
                "local-en": "all-MiniLM-L6-v2",  # 英文优化
            }

            actual_model = model_map.get(model_name, model_name)
            model = SentenceTransformer(actual_model, device=device)
            _local_models[model_name] = model
            logger.info(f"Loaded SentenceTransformer model: {actual_model} on {device}")
            return model

    except ImportError as e:
        logger.error(f"Failed to import local model library: {e}")
        raise ImportError(
            f"请安装本地模型依赖: pip install sentence-transformers FlagEmbedding"
        )
    except Exception as e:
        logger.error(f"Failed to load local model {model_name}: {e}")
        raise


def _run_local_model(model, texts: List[str]) -> List[List[float]]:
    """在线程池中运行本地模型（避免阻塞）"""
    # 判断模型类型
    if hasattr(model, 'encode'):
        # SentenceTransformer 或 M3E
        embeddings = model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
    elif hasattr(model, 'encode_queries'):
        # BGE/FlagEmbedding
        embeddings = model.encode_queries(texts)
        return embeddings.tolist()
    else:
        raise ValueError(f"Unknown model type: {type(model)}")


async def get_embeddings(
    texts: List[str],
    model: str = "local-zh",
    batch_size: int = 32,
) -> List[List[float]]:
    """
    获取文本的embeddings

    Args:
        texts: 文本列表
        model: 模型名称
              - 本地模型: "local", "local-zh", "local-en", "bge-small-zh-v1.5", "m3e-base" 等
              - 远程API: "embedding-3" (GLM), "text-embedding-3-small" (OpenAI)
        batch_size: 批处理大小

    Returns:
        embedding向量列表
    """
    # 判断是否使用本地模型
    is_local = not model.startswith("embedding") and not model.startswith("text-embedding")

    if is_local:
        # 使用本地模型
        try:
            local_model = _get_local_model(model)
            all_embeddings = []

            # 分批处理
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                # 在线程池中运行（避免阻塞事件循环）
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    _executor,
                    _run_local_model,
                    local_model,
                    batch
                )

                all_embeddings.extend(embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")

            logger.info(f"get_embeddings completed: {len(all_embeddings)} embeddings total")
            logger.info(f"Returning embeddings (first 3 dims: {len(all_embeddings[0]) if all_embeddings else 0} dimensions)...")
            return all_embeddings

        except Exception as e:
            logger.error(f"Failed to generate embeddings with local model: {e}")
            # 返回零向量
            zero_embedding = [0.0] * 768  # 大多数本地模型是768维
            return [zero_embedding] * len(texts)

    else:
        # 使用远程API (GLM/OpenAI等)
        from .llm import get_embedding_client

        # API 的 embedding 模型限制
        if model.startswith("embedding"):
            batch_size = min(batch_size, 64)  # GLM API 限制

        embedding_client = get_embedding_client(model)

        all_embeddings = []

        # 分批处理
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                embeddings = await embedding_client.aembed_documents(batch)
                all_embeddings.extend(embeddings)
                logger.info(f"Generated embeddings for batch {i//batch_size + 1} ({len(batch)} texts)")
            except Exception as e:
                logger.error(f"Failed to generate embeddings for batch: {e}")
                # 返回零向量作为fallback
                if model == "embedding-3":
                    zero_embedding = [0.0] * 1024
                else:
                    zero_embedding = [0.0] * 1536
                all_embeddings.extend([zero_embedding] * len(batch))

        logger.info(f"get_embeddings completed: {len(all_embeddings)} embeddings total")
        return all_embeddings


async def get_embedding(text: str, model: str = "local-zh") -> List[float]:
    """
    获取单个文本的embedding

    Args:
        text: 输入文本
        model: 模型名称

    Returns:
        embedding向量
    """
    embeddings = await get_embeddings([text], model=model)
    return embeddings[0]


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
