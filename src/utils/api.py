"""
API clients for academic databases
学术数据库API客户端
"""

import asyncio
import aiohttp
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class SemanticScholarAPI:
    """
    Semantic Scholar API客户端
    """

    BASE_URL = "https://api.semanticscholar.org/graph/v1"

    def __init__(self, api_key: Optional[str] = None):
        """
        初始化客户端

        Args:
            api_key: API密钥（可选，有更高的速率限制）
        """
        self.api_key = api_key
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_papers(
        self,
        query: str,
        year_range: Optional[tuple] = None,
        limit: int = 100,
        fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索论文

        Args:
            query: 搜索查询
            year_range: 年份范围 (start, end)
            limit: 返回数量限制
            fields: 返回字段列表

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # 默认字段
        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors", "year",
                "venue", "citationCount", "url", "doi", "publicationDate"
            ]

        # 构建请求参数
        params = {
            "query": query,
            "limit": min(limit, 100),  # S2 API单次最多100
            "fields": ",".join(fields),
        }

        # 添加年份过滤
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"

        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with self.session.get(
                f"{self.BASE_URL}/paper/search",
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()
                data = await response.json()

                papers = data.get("data", [])
                total = data.get("total", 0)

                logger.info(f"S2 API returned {len(papers)} papers (total: {total})")
                return papers

        except aiohttp.ClientError as e:
            logger.error(f"S2 API request failed: {e}")
            return []

    async def get_paper_details(
        self,
        paper_id: str,
        fields: Optional[List[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        获取论文详情

        Args:
            paper_id: 论文ID
            fields: 返回字段列表

        Returns:
            论文详情
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        if fields is None:
            fields = [
                "paperId", "title", "abstract", "authors", "year",
                "venue", "citationCount", "url", "doi", "references",
                "publicationTypes", "journal"
            ]

        params = {"fields": ",".join(fields)}
        headers = {}
        if self.api_key:
            headers["x-api-key"] = self.api_key

        try:
            async with self.session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params=params,
                headers=headers,
            ) as response:
                response.raise_for_status()
                return await response.json()

        except aiohttp.ClientError as e:
            logger.error(f"S2 API paper details request failed: {e}")
            return None


class ArxivAPI:
    """
    arXiv API客户端
    """

    BASE_URL = "http://export.arxiv.org/api/query"

    def __init__(self):
        """初始化客户端"""
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def search_papers(
        self,
        query: str,
        max_results: int = 100,
        sort_by: str = "relevance",
    ) -> List[Dict[str, Any]]:
        """
        搜索arXiv论文

        Args:
            query: 搜索查询
            max_results: 最大结果数
            sort_by: 排序方式

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # arXiv使用标准的search query语法
        # 例如: "cat:cs.AI AND deep learning"
        search_query = f"all:{query}"

        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": max_results,
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        try:
            async with self.session.get(
                self.BASE_URL,
                params=params,
            ) as response:
                response.raise_for_status()
                xml_content = await response.text()

                # 解析XML
                import xml.etree.ElementTree as ET
                root = ET.fromstring(xml_content)

                # 命名空间
                ns = {
                    "atom": "http://www.w3.org/2005/Atom",
                    "arxiv": "http://arxiv.org/schemas/atom",
                }

                papers = []
                for entry in root.findall("atom:entry", ns):
                    paper = {
                        "id": entry.find("atom:id", ns).text.split("/")[-1],
                        "title": entry.find("atom:title", ns).text.strip(),
                        "summary": entry.find("atom:summary", ns).text.strip(),
                        "published": entry.find("atom:published", ns).text,
                        "authors": [
                            author.find("atom:name", ns).text
                            for author in entry.findall("atom:author", ns)
                        ],
                        "categories": [
                            cat.get("term")
                            for cat in entry.findall("atom:category", ns)
                        ],
                        "url": entry.find("atom:id", ns).text,
                        "source": "arxiv",
                    }

                    # 尝试获取PDF链接
                    link = entry.find("atom:link[@title='pdf']", ns)
                    if link is not None:
                        paper["pdf_url"] = link.get("href")

                    papers.append(paper)

                logger.info(f"arXiv API returned {len(papers)} papers")
                return papers

        except (aiohttp.ClientError, Exception) as e:
            logger.error(f"arXiv API request failed: {e}")
            logger.error(f"Query was: {search_query}")
            return []


async def multi_source_search(
    queries: List[str],
    year_range: Optional[tuple] = None,
    max_results_per_source: int = 100,
) -> List[Dict[str, Any]]:
    """
    多源搜索

    Args:
        queries: 搜索查询列表
        year_range: 年份范围
        max_results_per_source: 每个源的最大结果数

    Returns:
        合并后的论文列表
    """
    all_papers = []

    # Semantic Scholar搜索
    async with SemanticScholarAPI() as s2_client:
        for query in queries:
            papers = await s2_client.search_papers(
                query=query,
                year_range=year_range,
                limit=max_results_per_source,
            )
            all_papers.extend(papers)

            # 速率限制
            await asyncio.sleep(0.1)

    # arXiv搜索
    async with ArxivAPI() as arxiv_client:
        for query in queries:
            papers = await arxiv_client.search_papers(
                query=query,
                max_results=max_results_per_source,
            )
            all_papers.extend(papers)

            await asyncio.sleep(0.1)

    logger.info(f"Multi-source search returned {len(all_papers)} total papers")
    return all_papers


def deduplicate_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    去重论文列表

    Args:
        papers: 论文列表

    Returns:
        去重后的论文列表
    """
    seen = set()
    unique_papers = []

    for paper in papers:
        # 使用title的标准化形式作为唯一标识
        title = paper.get("title", "").lower().strip()
        title_clean = "".join(c for c in title if c.isalnum())

        if title_clean and title_clean not in seen:
            seen.add(title_clean)
            unique_papers.append(paper)
        else:
            logger.debug(f"Duplicate paper removed: {title[:50]}")

    logger.info(f"Deduplicated: {len(papers)} -> {len(unique_papers)}")
    return unique_papers
