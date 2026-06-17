"""
Extended Academic Database APIs
扩展的学术论文数据库 API（替代 Semantic Scholar）

支持的数据库：
- arXiv: 计算机科学、物理、数学等
- PubMed: 医学、生物学、生命科学
- DBLP: 计算机科学文献
- Crossref: 跨学科开放获取文献
- Europe PMC: 生命科学、生物医学
"""

import asyncio
import aiohttp
import hashlib
from typing import Any, Dict, List, Optional
import logging
from datetime import datetime
import xml.etree.ElementTree as ET

logger = logging.getLogger(__name__)


class PubMedAPI:
    """
    PubMed API (免费，无需 API key)
    专注于医学、生物学、生命科学
    """

    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

    def __init__(self, api_key: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        """初始化客户端"""
        self.api_key = api_key
        self._external_session = session
        self.session: Optional[aiohttp.ClientSession] = session

    async def __aenter__(self):
        if self._external_session is None:
            self.session = aiohttp.ClientSession()
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
        year_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索 PubMed 论文

        Args:
            query: 搜索查询
            max_results: 最大结果数
            year_range: 年份范围

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # 第一步：搜索获取 PMIDs
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        search_params = {
            "db": "pubmed",
            "term": query,
            "retmax": str(max_results),
            "retmode": "json",
            "sort": "relevance",
        }

        # 添加年份过滤
        if year_range:
            search_params["datetype"] = "pubdate"
            search_params["reldate"] = f"{year_range[0]}:3000"  # 简化处理

        # 使用 API key 可以提高速率限制
        if self.api_key:
            search_params["api_key"] = self.api_key

        try:
            # 搜索获取 PMIDs
            async with self.session.get(search_url, params=search_params) as resp:
                if resp.status != 200:
                    logger.error(f"PubMed search failed: {resp.status}")
                    return []

                search_data = await resp.json()
                pmids = search_data.get("esearchresult", {}).get("idlist", [])

                if not pmids:
                    logger.info(f"PubMed: No results for query: {query}")
                    return []

                logger.info(f"PubMed: Found {len(pmids)} papers for query: {query}")

            # 第二步：获取详细信息
            fetch_url = f"{self.BASE_URL}/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids),
                "retmode": "xml",
            }

            if self.api_key:
                fetch_params["api_key"] = self.api_key

            async with self.session.get(fetch_url, params=fetch_params) as resp:
                if resp.status != 200:
                    logger.error(f"PubMed fetch failed: {resp.status}")
                    return []

                xml_content = await resp.text()

            # 解析 PubMed XML
            papers = await self._parse_pubmed_xml(xml_content)

            logger.info(f"PubMed: Retrieved {len(papers)} papers")
            return papers

        except Exception as e:
            logger.error(f"PubMed API request failed: {e}")
            return []

    async def _parse_pubmed_xml(self, xml_content: str) -> List[Dict[str, Any]]:
        """解析 PubMed XML 响应"""
        papers = []

        try:
            root = ET.fromstring(xml_content)

            # PubMed 使用默认命名空间
            articles = root.findall(".//PubmedArticle")

            for article in articles:
                try:
                    # PMID
                    pmid_elem = article.find(".//PMID")
                    pmid = pmid_elem.text if pmid_elem is not None else ""

                    # 标题
                    title_elem = article.find(".//ArticleTitle")
                    title = title_elem.text if title_elem is not None else "No title"

                    # 摘要
                    abstract_elem = article.find(".//AbstractText")
                    abstract = abstract_elem.text if abstract_elem is not None else ""

                    # 作者
                    authors = []
                    for author in article.findall(".//Author"):
                        last_name = author.find("LastName")
                        fore_name = author.find("ForeName")
                        if last_name is not None:
                            name = f"{last_name.text}"
                            if fore_name is not None:
                                name = f"{fore_name.text} {name}"
                            authors.append({"name": name})

                    # 期刊
                    journal_elem = article.find(".//Journal/Title")
                    journal = journal_elem.text if journal_elem is not None else ""

                    # 发表日期
                    pub_date_elem = article.find(".//PubDate/Year")
                    year = int(pub_date_elem.text) if pub_date_elem is not None else None

                    paper = {
                        "paperId": f"pubmed_{pmid}",
                        "id": f"pubmed_{pmid}",
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "authors": authors,
                        "venue": journal,
                        "source": "pubmed",
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    }

                    papers.append(paper)

                except Exception as e:
                    logger.debug(f"Failed to parse PubMed article: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse PubMed XML: {e}")

        return papers


class DBLPAPI:
    """
    DBLP API (免费，无需 API key)
    专注于计算机科学文献
    """

    BASE_URL = "https://dblp.org/search/publ/api"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """初始化客户端"""
        self._external_session = session
        self.session: Optional[aiohttp.ClientSession] = session

    async def __aenter__(self):
        if self._external_session is None:
            self.session = aiohttp.ClientSession()
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
    ) -> List[Dict[str, Any]]:
        """
        搜索 DBLP 论文

        Args:
            query: 搜索查询
            max_results: 最大结果数

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # DBLP API 参数
        params = {
            "q": query,
            "format": "json",
            "h": max_results,
        }

        try:
            async with self.session.get(self.BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"DBLP API failed: {resp.status}")
                    return []

                data = await resp.json()

                # 解析 DBLP 响应
                papers = await self._parse_dblp_json(data)

                logger.info(f"DBLP: Retrieved {len(papers)} papers")
                return papers

        except Exception as e:
            logger.error(f"DBLP API request failed: {e}")
            return []

    async def _parse_dblp_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析 DBLP JSON 响应"""
        papers = []

        try:
            result = data.get("result", {})
            hits = result.get("hits", {})

            for hit in hits.get("hit", []):
                entry = hit.get("info", {})

                # 标题
                title = entry.get("title", "")
                if not title:
                    continue

                # 作者
                authors = entry.get("authors", {})
                if isinstance(authors, dict):
                    author_list = authors.get("author", [])
                else:
                    author_list = authors if authors else []

                # 处理作者格式
                formatted_authors = []
                if isinstance(author_list, list):
                    for author in author_list[:10]:  # 限制作者数量
                        formatted_authors.append({"name": str(author)})

                # 年份
                year = None
                if "year" in entry:
                    try:
                        year = int(entry["year"])
                    except (ValueError, TypeError):
                        pass

                # 来源期刊
                venue = entry.get("venue", "")
                if isinstance(venue, dict):
                    venue = venue.get("text", str(venue))

                # URL
                url = entry.get("url", "")
                # DBLP 记录 key（稳定、可复现），用作去重主键
                key = entry.get("key", "")
                if not url and key:
                    url = f"https://dblp.org/rec/{key.replace('/', '.html')}"

                # 稳定 ID：优先用 DBLP key，否则用标题的 md5（内建 hash 跨进程不可复现）
                if key:
                    dblp_id = f"dblp_{key}"
                else:
                    dblp_id = "dblp_" + hashlib.md5(title.lower().encode()).hexdigest()[:16]

                paper = {
                    "paperId": dblp_id,
                    "id": dblp_id,
                    "title": title,
                    "abstract": "",  # DBLP 不提供摘要
                    "year": year,
                    "authors": formatted_authors,
                    "venue": venue,
                    "source": "dblp",
                    "url": url,
                }

                papers.append(paper)

        except Exception as e:
            logger.error(f"Failed to parse DBLP JSON: {e}")

        return papers


class EuropePMCAPI:
    """
    Europe PMC API (免费，无需 API key)
    生命科学和生物医学开放获取文献
    """

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, session: Optional[aiohttp.ClientSession] = None):
        """初始化客户端"""
        self._external_session = session
        self.session: Optional[aiohttp.ClientSession] = session

    async def __aenter__(self):
        if self._external_session is None:
            self.session = aiohttp.ClientSession()
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
        year_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索 Europe PMC 论文

        Args:
            query: 搜索查询
            max_results: 最大结果数
            year_range: 年份范围

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # 构建查询字符串
        query_string = query
        if year_range:
            # 添加年份过滤
            query_string += f' AND FIRST_PDATE:["{year_range[0]}-01-01" TO "{year_range[1]}-12-31"]'

        params = {
            "query": query_string,
            "resulttype": "core",
            "format": "json",
            "pageSize": str(min(max_results, 100)),
        }

        try:
            async with self.session.get(self.BASE_URL, params=params) as resp:
                if resp.status != 200:
                    logger.error(f"Europe PMC API failed: {resp.status}")
                    return []

                data = await resp.json()

                # 解析响应
                papers = await self._parse_epmc_json(data)

                logger.info(f"Europe PMC: Retrieved {len(papers)} papers")
                return papers

        except Exception as e:
            logger.error(f"Europe PMC API request failed: {e}")
            return []

    async def _parse_epmc_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析 Europe PMC JSON 响应"""
        papers = []

        try:
            result_list = data.get("resultList", {}).get("result", [])

            for item in result_list:
                try:
                    # 标题
                    title = item.get("title", "")
                    if not title:
                        continue

                    # 作者
                    authors = []
                    for author in item.get("authorString", "").split(",")[:10]:
                        authors.append({"name": author.strip()})

                    # 摘要
                    abstract = item.get("abstractText", "")

                    # 年份
                    year = None
                    pub_date = item.get("firstPublicationDate")
                    if pub_date:
                        try:
                            year = int(pub_date[:4])
                        except (ValueError, TypeError):
                            pass

                    # 来源期刊
                    journal = item.get("journalTitle", "")

                    # URL
                    url = item.get("pmcid", "")
                    if url:
                        url = f"https://europepmc.org/article/{url}"
                    else:
                        url = item.get("doi", "")
                        if url:
                            url = f"https://doi.org/{url}"

                    # DOI
                    doi = item.get("doi", "")

                    # 引用数
                    citations = item.get("citationCount", "0")
                    try:
                        citation_count = int(citations)
                    except (ValueError, TypeError):
                        citation_count = 0

                    # 稳定 ID：优先 Europe PMC 自带 id，否则用标题 md5
                    eid = item.get("id") or hashlib.md5(title.lower().encode()).hexdigest()[:16]
                    epmc_id = f"epmc_{eid}"

                    paper = {
                        "paperId": epmc_id,
                        "id": epmc_id,
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "authors": authors,
                        "venue": journal,
                        "source": "europe_pmc",
                        "url": url,
                        "doi": doi,
                        "citationCount": citation_count,
                    }

                    papers.append(paper)

                except Exception as e:
                    logger.debug(f"Failed to parse Europe PMC item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse Europe PMC JSON: {e}")

        return papers


class OpenAlexAPI:
    """
    OpenAlex API (完全免费，无需 API key)
    最大的开放学术引用数据库
    """

    BASE_URL = "https://api.openalex.org"

    @staticmethod
    def _inverted_index_to_text(inverted_index: Optional[Dict[str, List[int]]]) -> str:
        """将 OpenAlex 的 abstract_inverted_index 还原为文本"""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort(key=lambda x: x[0])
        return " ".join(w for _, w in word_positions)

    def __init__(self, email: Optional[str] = None, session: Optional[aiohttp.ClientSession] = None):
        """
        初始化客户端

        Args:
            email: 提供邮箱可以获得更高的速率限制（Polite Pool）
            session: 外部共享的 aiohttp session（可选）
        """
        self.email = email
        self._external_session = session
        self.session: Optional[aiohttp.ClientSession] = session

    async def __aenter__(self):
        if self._external_session is None:
            self.session = aiohttp.ClientSession()
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
        year_range: Optional[tuple] = None,
    ) -> List[Dict[str, Any]]:
        """
        搜索 OpenAlex 论文

        Args:
            query: 搜索查询
            max_results: 最大结果数
            year_range: 年份范围

        Returns:
            论文列表
        """
        if not self.session:
            raise RuntimeError("Session not initialized. Use 'async with' context.")

        # 构建过滤参数
        filters = []
        if year_range:
            filters.append(f"publication_year:{year_range[0]}-{year_range[1]}")

        filter_param = ",".join(filters) if filters else None

        params = {
            "search": query,
            "per-page": min(max_results, 200),
            "filter": filter_param,
        }

        headers = {}
        # 提供邮箱以获得更好的速率限制
        if self.email:
            headers["User-Agent"] = f"mailto:{self.email}"
        else:
            headers["User-Agent"] = "LiteratureReviewSystem/1.0"

        try:
            async with self.session.get(
                f"{self.BASE_URL}/works",
                params=params,
                headers=headers,
            ) as resp:
                if resp.status != 200:
                    logger.error(f"OpenAlex API failed: {resp.status}")
                    return []

                data = await resp.json()

                # 解析响应
                papers = await self._parse_openalex_json(data)

                logger.info(f"OpenAlex: Retrieved {len(papers)} papers")
                return papers

        except Exception as e:
            logger.error(f"OpenAlex API request failed: {e}")
            return []

    async def _parse_openalex_json(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """解析 OpenAlex JSON 响应"""
        papers = []

        try:
            results = data.get("results", [])
            meta = data.get("meta", {})

            for item in results:
                try:
                    # 标题
                    title = item.get("title", "")
                    if not title:
                        continue

                    # 作者
                    authors = []
                    for authorship in item.get("authorships", [])[:10]:
                        author = authorship.get("author", {})
                        if author:
                            name = author.get("display_name", "")
                            if name:
                                authors.append({"name": name})

                    # 摘要（从 abstract_inverted_index 还原）
                    abstract = self._inverted_index_to_text(
                        item.get("abstract_inverted_index")
                    )

                    # 年份
                    year = item.get("publication_year")
                    if year:
                        try:
                            year = int(year)
                        except (ValueError, TypeError):
                            year = None

                    # 来源期刊
                    venue = item.get("primary_location", {})
                    source = venue.get("source", {})
                    journal = source.get("display_name", "")

                    # URL
                    url = item.get("id", "")
                    if url:
                        url = f"https://openalex.org/{url}"

                    # DOI
                    doi = item.get("doi", "")

                    # 引用数
                    citations = item.get("cited_by_count", 0)

                    # 类型
                    type_ = item.get("type", "")

                    paper = {
                        "paperId": item.get("id", "").replace("/", "_"),
                        "id": item.get("id", "").replace("/", "_"),
                        "title": title,
                        "abstract": abstract,
                        "year": year,
                        "authors": authors,
                        "venue": journal,
                        "source": "openalex",
                        "url": url,
                        "doi": doi,
                        "citationCount": citations,
                        "type": type_,
                    }

                    papers.append(paper)

                except Exception as e:
                    logger.debug(f"Failed to parse OpenAlex item: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse OpenAlex JSON: {e}")

        return papers


# 便捷函数
async def search_all_databases(
    query: str,
    max_results: int = 100,
    year_range: Optional[tuple] = None,
    databases: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    从所有可用数据库并行搜索论文

    Args:
        query: 搜索查询
        max_results: 每个数据库的最大结果数
        year_range: 年份范围
        databases: 要搜索的数据库列表，None 表示全部

    Returns:
        合并后的论文列表
    """
    from .api import ArxivAPI

    if databases is None:
        databases = ["arxiv", "pubmed", "dblp", "europe_pmc", "openalex"]

    async def _search_arxiv(session):
        try:
            async with ArxivAPI(session=session) as api:
                return await api.search_papers(query=query, max_results=max_results)
        except Exception as e:
            logger.error(f"arXiv search failed: {e}")
            return []

    async def _search_pubmed(session):
        try:
            async with PubMedAPI(session=session) as api:
                return await api.search_papers(query=query, max_results=max_results, year_range=year_range)
        except Exception as e:
            logger.error(f"PubMed search failed: {e}")
            return []

    async def _search_dblp(session):
        try:
            async with DBLPAPI(session=session) as api:
                return await api.search_papers(query=query, max_results=max_results)
        except Exception as e:
            logger.error(f"DBLP search failed: {e}")
            return []

    async def _search_europe_pmc(session):
        try:
            async with EuropePMCAPI(session=session) as api:
                return await api.search_papers(query=query, max_results=max_results, year_range=year_range)
        except Exception as e:
            logger.error(f"Europe PMC search failed: {e}")
            return []

    async def _search_openalex(session):
        try:
            async with OpenAlexAPI(session=session) as api:
                return await api.search_papers(query=query, max_results=max_results, year_range=year_range)
        except Exception as e:
            logger.error(f"OpenAlex search failed: {e}")
            return []

    # 数据库搜索函数映射
    db_searchers = {
        "arxiv": _search_arxiv,
        "pubmed": _search_pubmed,
        "dblp": _search_dblp,
        "europe_pmc": _search_europe_pmc,
        "openalex": _search_openalex,
    }

    all_papers = []

    async with aiohttp.ClientSession() as shared_session:
        # 并行搜索所有目标数据库
        tasks = [db_searchers[db](shared_session) for db in databases if db in db_searchers]
        results = await asyncio.gather(*tasks)

        for papers in results:
            if isinstance(papers, list):
                all_papers.extend(papers)

    logger.info(f"All databases search returned {len(all_papers)} total papers")
    return all_papers
