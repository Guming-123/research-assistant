"""
Text processing utilities
文本处理工具
"""

import re
from typing import List, Optional
import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


async def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separator: str = "\n\n",
) -> List[str]:
    """
    将文本分块

    Args:
        text: 输入文本
        chunk_size: 块大小（字符数）
        chunk_overlap: 块重叠大小
        separator: 分隔符

    Returns:
        文本块列表
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[separator, "\n", ". ", " ", ""],
        length_function=len,
    )

    chunks = text_splitter.split_text(text)
    logger.debug(f"Chunked text into {len(chunks)} chunks")
    return chunks


def normalize_text(text: str) -> str:
    """
    标准化文本

    Args:
        text: 输入文本

    Returns:
        标准化后的文本
    """
    # 转小写
    text = text.lower()

    # 移除特殊字符（保留中文、字母、数字、基本标点）
    text = re.sub(r'[^\w\s\u4e00-\u9fff.,!?;:()\-\[\]{}"]', ' ', text)

    # 合并多个空白
    text = re.sub(r'\s+', ' ', text)

    # 移除URL
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)

    # 移除邮箱
    text = re.sub(r'\S+@\S+', '', text)

    return text.strip()


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    从文本中提取关键词（简单版）

    Args:
        text: 输入文本
        top_n: 返回前N个关键词

    Returns:
        关键词列表
    """
    # 简单的词频统计
    words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{2,}\b', text.lower())

    # 过滤常见停用词
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
        'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        '的', '了', '是', '在', '和', '与', '或', '但', '对于', '通过', '基于'
    }

    word_freq = {}
    for word in words:
        if word not in stopwords:
            word_freq[word] = word_freq.get(word, 0) + 1

    # 按频率排序
    sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def compute_normalized_frequency(
    chunks: List[str],
    retrieved_chunks: set,
) -> float:
    """
    计算归一化频率（NF）

    NF(d) = 被检索到的chunk数 / 总chunk数

    Args:
        chunks: 文档的所有chunk
        retrieved_chunks: 被检索到的chunk索引集合

    Returns:
        归一化频率值（0-1）
    """
    if not chunks:
        return 0.0

    return len(retrieved_chunks) / len(chunks)


def clean_abstract(abstract: str) -> str:
    """
    清理摘要文本

    Args:
        abstract: 原始摘要

    Returns:
        清理后的摘要
    """
    # 移除常见噪声
    patterns = [
        r'Abstract\s*:?\s*',
        r'摘要\s*:?\s*',
        r'\s*©\s*.*?\s*',
        r'\s*\d+\s*[A-Z]+\s*\d{4}\s*',  # 会议信息
    ]

    for pattern in patterns:
        abstract = re.sub(pattern, ' ', abstract, flags=re.IGNORECASE)

    # 清理空白
    abstract = re.sub(r'\s+', ' ', abstract)

    return abstract.strip()


def merge_overlapping_chunks(
    chunks: List[str],
    min_length: int = 100,
) -> List[str]:
    """
    合并过短的chunk

    Args:
        chunks: 原始chunk列表
        min_length: 最小长度

    Returns:
        合并后的chunk列表
    """
    merged = []
    current = ""

    for chunk in chunks:
        if len(chunk) < min_length:
            current += " " + chunk
        else:
            if current:
                merged.append(current.strip())
                current = ""
            merged.append(chunk)

    if current:
        merged.append(current.strip())

    return merged


def extract_sentences(text: str) -> List[str]:
    """
    从文本中提取句子

    Args:
        text: 输入文本

    Returns:
        句子列表
    """
    # 中英文句子分割
    sentence_endings = r'(?<=[.!?。！？])\s+'
    sentences = re.split(sentence_endings, text)
    return [s.strip() for s in sentences if s.strip()]


def truncate_text(text: str, max_length: int, suffix: str = "...") -> str:
    """
    截断文本到指定长度

    Args:
        text: 输入文本
        max_length: 最大长度
        suffix: 截断后缀

    Returns:
        截断后的文本
    """
    if len(text) <= max_length:
        return text

    return text[:max_length - len(suffix)] + suffix
