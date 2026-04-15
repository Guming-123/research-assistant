"""
Search Agent - 文献检索Agent
负责从学术数据库检索文献、去重、清洗、标准化
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult
from ..core.workspace import LiteratureRecord, SharedWorkspace
from ..utils.api import ArxivAPI, deduplicate_papers
from ..utils.api_extended import (
    PubMedAPI,
    DBLPAPI,
    EuropePMCAPI,
    OpenAlexAPI,
    search_all_databases,
)
from ..utils.pdf import extract_text_from_pdf
from ..utils.llm import get_llm_client
from ..utils.exceptions import SearchError, APIError, LLMError, ValidationError

logger = logging.getLogger(__name__)


class SearchAgent(BaseAgent):
    """
    文献检索Agent

    职责：
    ① 构建搜索查询
    ② 多源检索（Semantic Scholar, arXiv等）
    ③ 去重合并
    ④ 元数据标准化
    ⑤ PDF下载与文本提取
    """

    # 查询构建Prompt
    QUERY_BUILDING_PROMPT = """你是一个学术文献检索专家。给定以下研究主题和范围定义，
请生成结构化的搜索策略。

研究主题：{research_topic}
范围定义：{scope_description}
目标数据库：Semantic Scholar, arXiv

请输出JSON格式：
{{
  "search_queries": [
    {{
      "database": "数据库名",
      "query": "搜索查询字符串",
      "description": "查询说明"
    }}
  ],
  "expected_coverage": "预估覆盖范围说明",
  "keywords": ["关键词1", "关键词2", ...]
}}"""

    def __init__(
        self,
        workspace: SharedWorkspace,
        llm_client: Optional[ChatOpenAI] = None,
        config: Optional[AgentConfig] = None,
    ):
        """初始化Search Agent"""
        config = config or AgentConfig(
            name="SearchAgent",
            description="Searches and retrieves literature from academic databases",
            model="glm-4-flash",
            temperature=0.3,
        )
        super().__init__(config, workspace, llm_client)

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return "research_topic" in kwargs

    async def execute(self, **kwargs) -> AgentResult:
        """
        执行文献检索

        Args:
            research_topic: 研究主题
            scope_description: 范围描述（可选）
            year_range: 年份范围 (start, end)
            max_results: 最大结果数
            enable_pdf_download: 是否下载PDF

        Returns:
            AgentResult
        """
        research_topic = kwargs.get("research_topic")
        scope_description = kwargs.get("scope_description", f"Academic papers about {research_topic}")
        year_range = kwargs.get("year_range", (2018, 2025))
        max_results = kwargs.get("max_results", 500)
        enable_pdf_download = kwargs.get("enable_pdf_download", False)

        try:
            # ① 构建搜索查询
            self.log_progress("Building search queries...")
            queries = await self._build_search_queries(research_topic, scope_description)
            self.log_progress(f"Generated {len(queries)} search queries")

            # ② 多源检索
            self.log_progress("Searching academic databases...")
            raw_papers = await self._multi_source_search(queries, year_range, max_results)
            self.log_progress(f"Retrieved {len(raw_papers)} papers from databases")

            # ③ 去重合并
            self.log_progress("Deduplicating papers...")
            unique_papers = deduplicate_papers(raw_papers)
            self.log_progress(f"After deduplication: {len(unique_papers)} unique papers")

            # ④ 元数据标准化
            self.log_progress("Standardizing metadata...")
            literature_records = await self._standardize_records(unique_papers)

            # ⑤ 保存到工作区
            await self.workspace.add_literature(literature_records)
            self.log_progress(f"Saved {len(literature_records)} papers to workspace")

            # ⑥ PDF处理（可选）
            pdf_processed = 0
            if enable_pdf_download:
                self.log_progress("Downloading and processing PDFs...")
                pdf_processed = await self._process_pdfs(literature_records[:50])  # 限制处理数量

            metrics = {
                "total_retrieved": len(raw_papers),
                "unique_papers": len(unique_papers),
                "year_range": year_range,
                "pdf_processed": pdf_processed,
                "queries_used": len(queries),
            }

            return self._create_result(
                success=True,
                data={"papers_saved": len(literature_records)},
                metrics=metrics,
            )

        except Exception as e:
            error_msg = f"Search execution failed: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

    async def _build_search_queries(
        self,
        research_topic: str,
        scope_description: str,
    ) -> List[Dict[str, str]]:
        """
        使用LLM构建搜索查询

        Args:
            research_topic: 研究主题
            scope_description: 范围描述

        Returns:
            查询列表
        """
        try:
            messages = [
                SystemMessage(content="You are an academic literature search expert."),
                HumanMessage(
                    content=self.QUERY_BUILDING_PROMPT.format(
                        research_topic=research_topic,
                        scope_description=scope_description,
                    )
                ),
            ]

            response = await self._call_llm(messages, response_format="json")
            result = json.loads(response)

            queries = result.get("search_queries", [])

            # 如果LLM没有返回有效查询，使用默认策略
            if not queries:
                queries = self._default_search_queries(research_topic)

            return queries

        except Exception as e:
            self.log_progress(f"Query building failed: {e}, using defaults", "warning")
            return self._default_search_queries(research_topic)

    def _default_search_queries(self, research_topic: str) -> List[Dict[str, str]]:
        """默认搜索查询策略（使用多数据库）"""
        # 提取关键词
        keywords = research_topic.split()

        queries = []

        # 主要查询 - 使用多个数据库
        queries.append({
            "database": "multi",
            "query": research_topic,
            "description": "Multi-database search",
            "databases": ["arxiv", "pubmed", "openalex"],
        })

        # 关键词组合查询
        if len(keywords) > 1:
            # 只取前 2 个关键词组合
            for i in range(min(2, len(keywords) - 1)):
                query = f"{keywords[i]} {keywords[i + 1]}"
                queries.append({
                    "database": "multi",
                    "query": query,
                    "description": f"Keyword combination: {query}",
                    "databases": ["arxiv", "dblp", "openalex"],
                })

        return queries

    async def _multi_source_search(
        self,
        queries: List[Dict[str, str]],
        year_range: Tuple[int, int],
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """
        多源检索（使用新的数据库 API）

        Args:
            queries: 查询列表
            year_range: 年份范围
            max_results: 最大结果数

        Returns:
            论文列表
        """
        all_papers = []
        results_per_query = max(10, max_results // len(queries))

        self.log_progress(f"Executing {len(queries)} search queries ({results_per_query} papers each)...")

        for query_info in queries:
            query = query_info["query"]
            databases = query_info.get("databases", ["arxiv", "openalex"])

            self.log_progress(f"Query: '{query}' from {len(databases)} databases")

            # arXiv
            if "arxiv" in databases:
                try:
                    self.log_progress(f"  → arXiv: {query}")
                    async with ArxivAPI() as api:
                        papers = await api.search_papers(
                            query=query,
                            max_results=results_per_query,
                        )
                        all_papers.extend(papers)
                        self.log_progress(f"  ✓ arXiv: {len(papers)} papers")
                        await asyncio.sleep(0.5)
                except Exception as e:
                    self.log_progress(f"  ✗ arXiv failed: {e}", "warning")

            # PubMed
            if "pubmed" in databases:
                try:
                    self.log_progress(f"  → PubMed: {query}")
                    async with PubMedAPI() as api:
                        papers = await api.search_papers(
                            query=query,
                            max_results=results_per_query,
                            year_range=year_range,
                        )
                        all_papers.extend(papers)
                        self.log_progress(f"  ✓ PubMed: {len(papers)} papers")
                        await asyncio.sleep(0.5)
                except Exception as e:
                    self.log_progress(f"  ✗ PubMed failed: {e}", "warning")

            # DBLP
            if "dblp" in databases:
                try:
                    self.log_progress(f"  → DBLP: {query}")
                    async with DBLPAPI() as api:
                        papers = await api.search_papers(
                            query=query,
                            max_results=results_per_query,
                        )
                        all_papers.extend(papers)
                        self.log_progress(f"  ✓ DBLP: {len(papers)} papers")
                        await asyncio.sleep(0.5)
                except Exception as e:
                    self.log_progress(f"  ✗ DBLP failed: {e}", "warning")

            # Europe PMC
            if "europe_pmc" in databases:
                try:
                    self.log_progress(f"  → Europe PMC: {query}")
                    async with EuropePMCAPI() as api:
                        papers = await api.search_papers(
                            query=query,
                            max_results=results_per_query,
                            year_range=year_range,
                        )
                        all_papers.extend(papers)
                        self.log_progress(f"  ✓ Europe PMC: {len(papers)} papers")
                        await asyncio.sleep(0.5)
                except Exception as e:
                    self.log_progress(f"  ✗ Europe PMC failed: {e}", "warning")

            # OpenAlex
            if "openalex" in databases:
                try:
                    self.log_progress(f"  → OpenAlex: {query}")
                    async with OpenAlexAPI() as api:
                        papers = await api.search_papers(
                            query=query,
                            max_results=results_per_query,
                            year_range=year_range,
                        )
                        all_papers.extend(papers)
                        self.log_progress(f"  ✓ OpenAlex: {len(papers)} papers")
                        await asyncio.sleep(0.5)
                except Exception as e:
                    self.log_progress(f"  ✗ OpenAlex failed: {e}", "warning")

            # 延迟避免过度请求
            await asyncio.sleep(1.0)

        self.log_progress(f"Total papers retrieved: {len(all_papers)}")
        return all_papers

    async def _standardize_records(
        self,
        papers: List[Dict[str, Any]],
    ) -> List[LiteratureRecord]:
        """
        标准化为LiteratureRecord格式

        Args:
            papers: 原始论文数据

        Returns:
            LiteratureRecord列表
        """
        records = []

        for paper in papers:
            try:
                # 处理作者
                authors = []
                if "authors" in paper:
                    if isinstance(paper["authors"], list):
                        authors = [
                            author.get("name", "")
                            if isinstance(author, dict) else str(author)
                            for author in paper["authors"]
                        ]
                    else:
                        authors = [str(paper["authors"])]

                # 处理年份
                year = paper.get("year")
                if year is None and "publicationDate" in paper:
                    year = int(paper["publicationDate"][:4])

                # 处理摘要
                abstract = paper.get("abstract") or paper.get("summary", "")

                # 处理来源
                source = paper.get("source", "semantic_scholar")
                if "arxiv" in paper.get("url", "").lower():
                    source = "arxiv"

                record = LiteratureRecord(
                    id=paper.get("paperId") or paper.get("id", ""),
                    title=paper.get("title", "").strip(),
                    authors=authors,
                    abstract=abstract,
                    year=year or 0,
                    source=source,
                    url=paper.get("url", ""),
                    doi=paper.get("doi"),
                    venue=paper.get("venue"),
                    citation_count=paper.get("citationCount"),
                    keywords=paper.get("keywords", []),
                )

                records.append(record)

            except Exception as e:
                self.log_progress(f"Failed to standardize record: {e}", "warning")
                continue

        return records

    async def _process_pdfs(
        self,
        records: List[LiteratureRecord],
        pdf_dir: str = "./workspace/pdfs",
    ) -> int:
        """
        下载和处理PDF

        Args:
            records: 文献记录列表
            pdf_dir: PDF保存目录

        Returns:
            成功处理的数量
        """
        pdf_path = Path(pdf_dir)
        pdf_path.mkdir(parents=True, exist_ok=True)

        processed = 0

        for record in records:
            try:
                # arXiv直接下载链接
                if record.source == "arxiv" and record.url:
                    pdf_url = record.url.replace("/abs/", "/pdf/") + ".pdf"
                # 其他来源使用DOI
                elif record.doi:
                    pdf_url = f"https://doi.org/{record.doi}"
                else:
                    continue

                # 下载PDF（这里简化处理，实际应该使用aiohttp下载）
                # 由于PDF下载较复杂，这里只记录PDF路径
                pdf_filename = f"{record.id or hash(record)}.pdf"
                record.pdf_path = str(pdf_path / pdf_filename)

                # 如果需要实际下载，可以在这里添加下载逻辑
                # 并使用 extract_text_from_pdf 提取文本

                processed += 1

            except Exception as e:
                self.log_progress(f"Failed to process PDF for {record.title[:30]}: {e}", "warning")
                continue

        return processed

    async def get_search_statistics(self) -> Dict[str, Any]:
        """获取搜索统计信息"""
        papers = await self.workspace.get_literature()

        if not papers:
            return {"total": 0}

        # 统计来源分布
        source_dist = {}
        year_dist = {}
        venue_dist = {}

        for paper in papers:
            # 来源分布
            source_dist[paper.source] = source_dist.get(paper.source, 0) + 1

            # 年份分布
            if paper.year:
                year_dist[paper.year] = year_dist.get(paper.year, 0) + 1

            # 期刊/会议分布
            if paper.venue:
                venue_dist[paper.venue] = venue_dist.get(paper.venue, 0) + 1

        return {
            "total": len(papers),
            "sources": source_dist,
            "years": dict(sorted(year_dist.items())),
            "venues": dict(sorted(venue_dist.items(), key=lambda x: x[1], reverse=True)[:10]),
        }
