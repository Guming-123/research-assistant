"""
Summary Agent - 综述生成Agent
负责对每个主题簇生成结构化摘要，最终整合为完整综述报告
"""

import asyncio
import json
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult
from ..core.workspace import ClusterResult, SharedWorkspace
from ..core.rq_manager import RQTree, RQLevel
from ..utils.llm import get_llm_client

logger = logging.getLogger(__name__)


class SummaryAgent(BaseAgent):
    """
    综述生成Agent

    职责：
    ① 按簇提取论文内容
    ② RAG摘要生成（二级RQ）
    ③ 深度分析（三级RQ）
    ④ 跨簇趋势综合
    ⑤ 生成最终报告
    """

    # 二级RQ摘要Prompt（方法论维度）
    METHODOLOGY_PROMPT = """你是一个学术综述撰写专家。请基于以下论文簇的内容，
回答研究问题。

研究问题（RQ11）：该簇中的论文使用了哪些方法论？

簇标签：{cluster_label}
簇描述：{cluster_description}

簇内代表性论文：
{papers_summary}

请按以下结构输出（中文，学术风格）：

## 方法概述
该簇涉及的主要方法类型（列表）

## 技术细节
每种方法的核心技术要点（2-3句/方法）

## 方法对比
| 方法 | 优势 | 局限 |
|------|------|------|
| ... | ... | ... |

## 演进趋势
方法随时间的变化趋势

## 研究空白
当前方法尚未解决的问题
"""

    # 二级RQ摘要Prompt（应用维度）
    APPLICATION_PROMPT = """研究问题（RQ21）：这些方法被应用在哪些领域？

簇标签：{cluster_label}
簇描述：{cluster_description}

簇内代表性论文：
{papers_summary}

请按以下结构输出（中文，学术风格）：

## 应用领域列表
该簇论文涉及的应用领域

## 领域-方法映射
| 领域 | 主要方法 | 典型应用 |
|------|----------|----------|
| ... | ... | ... |

## 领域挑战
每个领域面临的主要挑战和局限

## 跨领域机会
不同领域之间可能的交叉机会
"""

    # 最终综述整合Prompt
    FINAL_REPORT_PROMPT = """你是一个资深学术综述撰写专家。基于以下各主题簇的分析结果，
请撰写一份完整的文献综述报告。

研究主题：{research_topic}
综述时间范围：{year_range}
分析论文总数：{total_papers}

各簇摘要：
{cluster_summaries}

请按以下结构撰写（中文，学术风格）：

# {research_topic}研究综述

## 1. 引言
### 1.1 研究背景与意义
### 1.2 综述范围与方法论说明

## 2. 方法论综述
{methodology_sections}

## 3. 应用领域综述
{application_sections}

## 4. 研究趋势与演进
### 4.1 方法论演进时间线
### 4.2 应用领域扩展趋势
### 4.3 跨学科融合趋势

## 5. 挑战与未来方向
### 5.1 当前研究的共性局限
### 5.2 未来可能的突破方向
### 5.3 对从业者的建议

## 6. 结论

要求：
- 每个论点需引用具体论文支撑（使用[作者, 年份]格式）
- 对比分析要客观、有数据支撑
- 语言精炼，避免空洞表述
- 使用学术化语言
"""

    def __init__(
        self,
        workspace: SharedWorkspace,
        llm_client: Optional[ChatOpenAI] = None,
        config: Optional[AgentConfig] = None,
    ):
        """初始化Summary Agent"""
        config = config or AgentConfig(
            name="SummaryAgent",
            description="Generates literature review summaries",
            model="glm-4-plus",
            temperature=0.5,
        )
        super().__init__(config, workspace, llm_client)

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return "rq_tree" in kwargs

    async def execute(self, **kwargs) -> AgentResult:
        """
        执行综述生成

        Args:
            rq_tree: RQ树结构
            include_methodology: 是否包含方法论分析
            include_applications: 是否包含应用分析

        Returns:
            AgentResult
        """
        rq_tree = kwargs.get("rq_tree")
        include_methodology = kwargs.get("include_methodology", True)
        include_applications = kwargs.get("include_applications", True)

        try:
            # 获取聚类结果
            clusters = await self.workspace.get_clusters()
            if not clusters:
                return self._create_result(
                    success=False,
                    errors=["No clusters found in workspace"]
                )

            self.log_progress(f"Generating summaries for {len(clusters)} clusters...")

            # ① 按簇提取论文内容
            cluster_data = await self._extract_cluster_content(clusters)

            # ② 生成各簇摘要
            cluster_summaries = []
            for cluster in clusters:
                self.log_progress(f"Processing cluster {cluster.cluster_id}: {cluster.label}")

                summary = await self._generate_cluster_summary(
                    cluster,
                    cluster_data[cluster.cluster_id],
                    include_methodology,
                    include_applications,
                )
                cluster_summaries.append(summary)

            # ③ 生成最终报告
            self.log_progress("Generating final literature review report...")
            final_report = await self._generate_final_report(
                rq_tree,
                cluster_summaries,
                cluster_data,
            )

            # 保存报告
            report_path = await self._save_report(final_report)

            # 保存各簇摘要
            for i, summary in enumerate(cluster_summaries):
                await self.workspace.save_summary(
                    f"cluster_{clusters[i].cluster_id}",
                    summary["content"],
                )

            metrics = {
                "cluster_count": len(clusters),
                "total_paper_count": sum(len(c["papers"]) for c in cluster_data.values()),
                "report_path": report_path,
                "word_count": len(final_report.split()),
            }

            self.log_progress(f"Report generation complete: {report_path}")

            return self._create_result(
                success=True,
                data={"report_path": report_path, "summary_count": len(cluster_summaries)},
                metrics=metrics,
            )

        except Exception as e:
            error_msg = f"Summary execution failed: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

    async def _extract_cluster_content(
        self,
        clusters: List[ClusterResult],
    ) -> Dict[int, Dict[str, Any]]:
        """
        提取各簇的论文内容

        Args:
            clusters: 簇列表

        Returns:
            {cluster_id: {papers, summaries, ...}}
        """
        cluster_data = {}

        for cluster in clusters:
            papers = await self.workspace.get_cluster_papers(cluster.cluster_id)

            # 准备论文摘要
            papers_summary = []
            for paper in papers[:10]:  # 限制数量
                papers_summary.append({
                    "title": paper.title,
                    "authors": paper.authors[:3] if len(paper.authors) > 3 else paper.authors,
                    "year": paper.year,
                    "abstract": (paper.abstract or "")[:300],
                    "venue": paper.venue,
                })

            cluster_data[cluster.cluster_id] = {
                "papers": papers,
                "papers_summary": papers_summary,
                "cluster": cluster,
            }

        return cluster_data

    async def _generate_cluster_summary(
        self,
        cluster: ClusterResult,
        data: Dict[str, Any],
        include_methodology: bool,
        include_applications: bool,
    ) -> Dict[str, str]:
        """
        生成单个簇的摘要

        Args:
            cluster: 簇信息
            data: 簇数据
            include_methodology: 是否包含方法论分析
            include_applications: 是否包含应用分析

        Returns:
            摘要内容字典
        """
        content_parts = []

        # 簇标题和描述
        content_parts.append(f"## {cluster.label}\n")
        content_parts.append(f"{cluster.description}\n")

        papers_summary_text = "\n".join([
            f"- [{p['year']}] {p['title']}\n  {p['abstract']}"
            for p in data["papers_summary"]
        ])

        # 方法论分析
        if include_methodology:
            try:
                messages = [
                    SystemMessage(content="You are an academic literature review expert."),
                    HumanMessage(
                        content=self.METHODOLOGY_PROMPT.format(
                            cluster_label=cluster.label,
                            cluster_description=cluster.description,
                            papers_summary=papers_summary_text,
                        )
                    ),
                ]

                response = await self._call_llm(messages)
                content_parts.append("### 方法论分析\n")
                content_parts.append(response)
                content_parts.append("\n")

            except Exception as e:
                self.log_progress(f"Methodology analysis failed for cluster {cluster.cluster_id}: {e}", "warning")

        # 应用分析
        if include_applications:
            try:
                messages = [
                    SystemMessage(content="You are an academic literature review expert."),
                    HumanMessage(
                        content=self.APPLICATION_PROMPT.format(
                            cluster_label=cluster.label,
                            cluster_description=cluster.description,
                            papers_summary=papers_summary_text,
                        )
                    ),
                ]

                response = await self._call_llm(messages)
                content_parts.append("### 应用分析\n")
                content_parts.append(response)
                content_parts.append("\n")

            except Exception as e:
                self.log_progress(f"Application analysis failed for cluster {cluster.cluster_id}: {e}", "warning")

        return {
            "cluster_id": cluster.cluster_id,
            "label": cluster.label,
            "content": "\n".join(content_parts),
        }

    async def _generate_final_report(
        self,
        rq_tree: RQTree,
        cluster_summaries: List[Dict[str, str]],
        cluster_data: Dict[int, Dict[str, Any]],
    ) -> str:
        """
        生成最终综述报告

        Args:
            rq_tree: RQ树
            cluster_summaries: 各簇摘要
            cluster_data: 簇数据

        Returns:
            完整报告文本
        """
        # 准备各簇摘要
        cluster_summaries_text = "\n\n---\n\n".join([
            f"### Cluster {s['cluster_id']}: {s['label']}\n{s['content']}"
            for s in cluster_summaries
        ])

        # 准备方法论章节
        methodology_sections = "\n".join([
            f"### 2.{i+1} {s['label']}\n{s['content'][:500]}..."
            for i, s in enumerate(cluster_summaries)
        ])

        # 准备应用章节
        application_sections = methodology_sections  # 简化处理

        # 估计年份范围
        all_years = []
        for data in cluster_data.values():
            all_years.extend([p.year for p in data["papers"] if p.year])
        year_range = (
            f"{min(all_years) if all_years else 'N/A'}-{max(all_years) if all_years else 'N/A'}"
        )

        total_papers = sum(len(data["papers"]) for data in cluster_data.values())

        # 生成最终报告
        messages = [
            SystemMessage(content="You are a senior academic literature review expert."),
            HumanMessage(
                content=self.FINAL_REPORT_PROMPT.format(
                    research_topic=rq_tree.research_topic,
                    year_range=year_range,
                    total_papers=total_papers,
                    cluster_summaries=cluster_summaries_text,
                    methodology_sections=methodology_sections,
                    application_sections=application_sections,
                )
            ),
        ]

        try:
            response = await self._call_llm(messages)
            return response
        except Exception as e:
            self.log_progress(f"Final report generation failed: {e}", "warning")
            # Fallback: 简单拼接各簇摘要
            return self._fallback_report(rq_tree, cluster_summaries, all_years, total_papers)

    def _fallback_report(
        self,
        rq_tree: RQTree,
        cluster_summaries: List[Dict[str, str]],
        all_years: List[int],
        total_papers: int,
    ) -> str:
        """
        备用报告生成

        Args:
            rq_tree: RQ树
            cluster_summaries: 各簇摘要
            all_years: 所有年份
            total_papers: 总论文数

        Returns:
            报告文本
        """
        year_range = (
            f"{min(all_years) if all_years else 'N/A'}-{max(all_years) if all_years else 'N/A'}"
        )

        report_parts = [
            f"# {rq_tree.research_topic}研究综述\n",
            "## 1. 引言\n",
            f"本综述针对{rq_tree.research_topic}领域进行了系统性的文献分析。",
            f"综述涵盖了{year_range}年间共{total_papers}篇相关论文。\n",
            "## 2. 主要主题\n",
        ]

        for i, summary in enumerate(cluster_summaries, 1):
            report_parts.append(f"### 2.{i} {summary['label']}\n")
            report_parts.append(summary['content'])
            report_parts.append("\n")

        report_parts.extend([
            "## 3. 结论\n",
            "本综述总结了该领域的主要研究方向和进展，为后续研究提供了参考。",
        ])

        return "\n".join(report_parts)

    async def _save_report(self, report: str) -> str:
        """
        保存报告

        Args:
            report: 报告内容

        Returns:
            保存路径
        """
        reports_dir = Path(self.workspace.base_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"literature_review_{timestamp}.md"

        async with report_path.open("w", encoding="utf-8") as f:
            await f.write(report)

        return str(report_path)

    async def get_report_summary(self) -> Optional[Dict[str, Any]]:
        """
        获取报告摘要

        Returns:
            报告摘要信息
        """
        summaries = await self.workspace.get_all_summaries()

        return {
            "cluster_summaries": len([k for k in summaries.keys() if k.startswith("cluster_")]),
            "summaries": list(summaries.keys()),
        }
