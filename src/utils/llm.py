"""
LLM Client utilities
提供统一的LLM客户端访问接口
"""

import os
from typing import Optional, List, Any
import logging

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


# 全局LLM客户端缓存
_llm_clients: dict = {}
_embedding_clients: dict = {}


def get_llm_client(
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    **kwargs
) -> ChatOpenAI:
    """
    获取LLM客户端（带缓存）

    Args:
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        api_key: API密钥
        base_url: API基础URL
        **kwargs: 其他参数

    Returns:
        ChatOpenAI实例
    """
    cache_key = f"{model}_{temperature}_{max_tokens}"

    if cache_key in _llm_clients:
        return _llm_clients[cache_key]

    # 获取配置
    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    client = ChatOpenAI(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        api_key=api_key,
        base_url=base_url,
        **kwargs
    )

    _llm_clients[cache_key] = client
    logger.info(f"Created LLM client: {model}")
    return client


def get_embedding_client(
    model: str = "embedding-3",
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
) -> OpenAIEmbeddings:
    """
    获取Embedding客户端（带缓存）

    Args:
        model: 模型名称（GLM: embedding-3, OpenAI: text-embedding-3-small）
        api_key: API密钥
        base_url: API基础URL

    Returns:
        OpenAIEmbeddings实例
    """
    if model in _embedding_clients:
        return _embedding_clients[model]

    api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_API_KEY")
    base_url = base_url or os.getenv("OPENAI_BASE_URL")

    if not api_key:
        raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

    client = OpenAIEmbeddings(
        model=model,
        api_key=api_key,
        base_url=base_url,
    )

    _embedding_clients[model] = client
    logger.info(f"Created embedding client: {model}")
    return client


async def invoke_llm(
    messages: List[BaseMessage],
    model: str = "gpt-4o",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    response_format: Optional[str] = None,
    **kwargs
) -> str:
    """
    调用LLM的便捷函数

    Args:
        messages: 消息列表
        model: 模型名称
        temperature: 温度参数
        max_tokens: 最大token数
        response_format: 响应格式
        **kwargs: 其他参数

    Returns:
        LLM响应
    """
    llm = get_llm_client(model, temperature, max_tokens)

    if response_format == "json":
        kwargs["response_format"] = {"type": "json_object"}

    response = await llm.ainvoke(messages, **kwargs)
    return response.content


def clear_client_cache() -> None:
    """清除客户端缓存"""
    global _llm_clients, _embedding_clients
    _llm_clients.clear()
    _embedding_clients.clear()
    logger.info("Cleared LLM client cache")
