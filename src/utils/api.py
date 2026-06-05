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
    arXiv API客户端（支持HTTPS、超时、重试）
    """

    BASE_URL = "https://export.arxiv.org/api/query"

    def __init__(self, timeout: int = 30, max_retries: int = 1, session: Optional[aiohttp.ClientSession] = None):
        """初始化客户端"""
        self._external_session = session
        self.session: Optional[aiohttp.ClientSession] = session
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_retries = max_retries

    async def __aenter__(self):
        if self._external_session is None:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        else:
            self.session = self._external_session
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._external_session is None and self.session:
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
            "max_results": min(max_results, 100),
            "sortBy": sort_by,
            "sortOrder": "descending",
        }

        import xml.etree.ElementTree as ET

        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.get(
                    self.BASE_URL,
                    params=params,
                ) as response:
                    response.raise_for_status()
                    xml_content = await response.text()

                    # 解析XML
                    root = ET.fromstring(xml_content)

                    # 命名空间
                    ns = {
                        "atom": "http://www.w3.org/2005/Atom",
                        "arxiv": "http://arxiv.org/schemas/atom",
                    }

                    papers = []
                    for entry in root.findall("atom:entry", ns):
                        # 跳过错误条目（arXiv API 有时返回错误entry）
                        title_el = entry.find("atom:title", ns)
                        if title_el is None:
                            continue

                        paper = {
                            "id": entry.find("atom:id", ns).text.split("/")[-1],
                            "title": title_el.text.strip(),
                            "summary": (entry.find("atom:summary", ns).text or "").strip(),
                            "published": entry.find("atom:published", ns).text,
                            "authors": [
                                author.find("atom:name", ns).text
                                for author in entry.findall("atom:author", ns)
                                if author.find("atom:name", ns) is not None
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

            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                logger.warning(f"arXiv API attempt {attempt + 1}/{self.max_retries + 1} failed: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(3.0 * (attempt + 1))
                else:
                    logger.error(f"arXiv API all retries exhausted. Query: {search_query}")
                    return []
            except ET.ParseError as e:
                logger.error(f"arXiv XML parse error: {e}")
                return []
            except Exception as e:
                logger.error(f"arXiv API unexpected error: {e}")
                return []

        return []


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
