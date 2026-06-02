"""
Summary Agent - 综述生成Agent
负责对每个主题簇生成结构化摘要，最终整合为完整综述报告
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from datetime import datetime
from pathlib import Path
import logging

import aiofiles

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult, get_agent_model, get_config
from ..core.workspace import ClusterResult, SharedWorkspace
from ..core.rq_manager import RQTree, RQLevel
from ..utils.llm import get_llm_client

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """检测文本主要语言，返回 'zh' 或 'en'"""
    zh_chars = len(re.findall(r'[一-鿿]', text))
    en_chars = len(re.findall(r'[a-zA-Z]', text))
    return 'zh' if zh_chars > en_chars else 'en'


# ──── 语言约束指令（会被嵌入所有 prompt） ────
_LANG_CONSTRAINT_ZH = (
    "【语言硬性要求】全文必须使用中文撰写（论文引用中的英文标题、人名、公式除外）。"
    "严禁中英文混用，禁止在正文叙述中插入英文句子或英文术语（除非引用原文）。"
    "专业术语可附英文缩写，但正文必须是中文。"
)

_LANG_CONSTRAINT_EN = (
    "【Language Requirement】The entire report MUST be written in English "
    "(cited paper titles in their original language are exempted). "
    "Do NOT mix Chinese and English in the same paragraph. "
    "Use English consistently throughout the main text."
)


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

    METHODOLOGY_PROMPT = """{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.
Your core principle: **Focus ONLY on underlying principles, core formulas, and mathematical derivations. Reject vague generalities.**

Analyze the methodology in depth based on the following cluster of papers.

Research Question (RQ11): What methodologies are used in this cluster?

Cluster Label: {cluster_label}
Cluster Description: {cluster_description}

Representative Papers in the Cluster:
{papers_summary}

【Absolute Requirements - Each violation makes the review worthless】
1. Every method MUST be accompanied by its core formula(s), with EVERY symbol explained (meaning, physical dimension, typical value range)
2. Every formula MUST state its derivation source (from which physical law, theorem, or prior formula it is derived)
3. Every performance claim MUST cite specific quantitative data (exact numbers, not "good" or "significant")
4. Every comparison MUST explain WHY at the formula/physics level (not "A is better than B" but "A outperforms B because A's formula captures X while B's formula neglects Y")

【Strictly Prohibited】
- Listing method names without their core formulas and derivation paths
- "This method shows significant results" without specific metric values and formula-based explanation
- "A is better than B" without explaining the formula/physics reason for the difference
- Omitting key formulas, equations, or physical parameters
- Summarizing a method's principle in a single sentence like "utilizes XX technique"
- Fabricating data, formulas, or methods not present in the papers

【Citation Format - Must Strictly Follow】
Citations must include author, year, and paper title:
Correct: Zhang et al. proposed a new method [Zhang, 2023, A Novel Method for XX]
Correct: This method was first proposed in [Li, 2021, Deep Learning Based Optimization]
Wrong: [Zhang, 2023] ← missing title
Wrong: [1] ← missing author, year, and title
Do NOT cite papers not listed above in "Representative Papers"

Please output in the following structure:

## 1. Core Formula Interpretation
For EACH method in this cluster:
### 1.x [Method Name]
- **Core formula**: Write out the key equation(s). Explain EVERY symbol: its meaning, physical dimension, and typical value range
- **Formula origin**: Which physical law, theorem, or prior equation does this formula derive from? Show the derivation chain
- **Assumptions and validity**: Under what conditions does this formula hold? When does it break down?
- **Parameter sensitivity**: Which parameter has the greatest impact on the output? Explain WHY from the formula structure (e.g., "parameter α appears in the exponent, making the output exponentially sensitive to small changes")

## 2. Derivation Chain Analysis
For the most important formulas:
- Show the complete derivation path from basic assumptions/axioms to the final result
- Identify which step introduces the most critical approximation and why
- Explain what physical insight is gained at each derivation step

## 3. Quantitative Comparison
| Method | Core Formula | Key Parameters (with ranges) | Reported Metric Values | Theoretical Limit (derived from formula) | Gap to Theoretical Limit |
|--------|-------------|------------------------------|----------------------|---------------------------------------|------------------------|

## 4. Performance Bottleneck Root Causes
What is the core bottleneck? Explain from the FORMULA perspective:
- Which term in the formula creates the bottleneck? Why?
- Is the bottleneck from a fundamental physical law (irreducible) or from an engineering approximation (potentially improvable)?
- What would need to change in the formula to break through?

## 5. Principle-Level Evolution Logic
For each step of method evolution:
- What specific formula/physics limitation did the new method address?
- What new formula or assumption did it introduce?
- What new constraints did this create? (traceable to specific formula terms)

## 6. Research Gaps and Opportunities
From a FORMULA and PRINCIPLE perspective:
- Which formula terms are currently approximated that could be made exact?
- What overlooked physical effects are not captured in current formulas?
- What cross-domain formulas might be transferable, and under what conditions?
"""

    APPLICATION_PROMPT = """{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.

Research Question (RQ21): In which fields are these methods applied, and what are the underlying principle-level reasons for their suitability?

Cluster Label: {cluster_label}
Cluster Description: {cluster_description}

Representative Papers in the Cluster:
{papers_summary}

【Absolute Requirements】
1. For each application, explain WHY the method is suitable from a formula/physics perspective (not just "it works well")
2. Show how the core formula adapts or transforms for different application scenarios
3. Explain performance differences across scenarios from the formula/parameter perspective
4. Distinguish between physical limitations (from fundamental laws) and engineering limitations (from manufacturing/process constraints)

【Strictly Prohibited】
- Listing application names without explaining how the method's formulas/models work in that specific application
- "This method can be widely applied in multiple fields" and other vague statements
- Ignoring specific challenges and failure cases, especially their formula-level causes
- Describing all fields as "having broad prospects"
- Fabricating non-existent application cases
- Summarizing applicability as "suitable due to high accuracy" without formula-based reasoning

【Citation Format - Must Strictly Follow】
Correct: Zhang et al. proposed a new method [Zhang, 2023, A Novel Method for XX]
Correct: This method was first proposed in [Li, 2021, Deep Learning Based Optimization]
Wrong: [Zhang, 2023] ← missing title
Wrong: [1] ← missing author, year, and title
Do NOT cite papers not listed above

Please output in the following structure:

## 1. Application Fields and Principle-Level Suitability
For each application field:
- Which specific formula/physics property makes the method suitable? (e.g., "The method's formula f(x) = α·exp(-βx) naturally models the decay process in X field because...")
- How do the method's key parameters map to application-specific quantities?

## 2. Formula Adaptation Across Scenarios
| Field | Core Formula Adaptation | Key Parameter Changes | Performance Impact |
|-------|------------------------|----------------------|-------------------|
Show how the core formula is modified, which parameters change, and the quantitative impact

## 3. Performance Differences: Principle-Level Attribution
When the same method performs differently across fields:
- Which formula term causes the difference? (e.g., "In field A, parameter β≈0.1 so the exponential term is negligible, but in field B, β≈10 causing the term to dominate")
- Is the performance gap due to the formula's assumptions being violated, or parameter values being unfavorable?

## 4. Field Challenges: Physical vs. Engineering Limitations
For each field's main challenge:
- **Physical limitation**: Constraints from fundamental laws (e.g., thermodynamic limits, quantum limits). These are irreducible.
- **Engineering limitation**: Constraints from current manufacturing, materials, or process capabilities. These are potentially improvable.
- Which formula term represents each limitation?

## 5. Cross-field Principle Transfer Opportunities
- Which formulas/physics principles from one field could transfer to another?
- What conditions must the target field satisfy for the transfer to be valid? (expressed as parameter ranges or formula assumptions)
- Which cross-domain formula combinations might lead to breakthroughs?
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
            model=get_agent_model("summary"),
            temperature=0.5,
            max_tokens=get_config().get("agents", {}).get("summary", {}).get("max_tokens", 16384),
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

            # 检测研究主题语言，生成语言约束指令
            topic = rq_tree.research_topic
            lang = _detect_language(topic)
            lang_constraint = _LANG_CONSTRAINT_ZH if lang == 'zh' else _LANG_CONSTRAINT_EN
            self.log_progress(f"Detected topic language: {lang}, will enforce consistent language")

            # ① 按簇提取论文内容
            cluster_data = await self._extract_cluster_content(clusters)

            # ② 并行生成各簇摘要
            self.log_progress(f"Generating summaries for {len(clusters)} clusters in parallel...")

            async def _gen_summary(cluster: ClusterResult) -> Dict[str, str]:
                summary = await self._generate_cluster_summary(
                    cluster,
                    cluster_data[cluster.cluster_id],
                    include_methodology,
                    include_applications,
                    lang_constraint=lang_constraint,
                )
                self.log_progress(f"Done cluster {cluster.cluster_id}: {cluster.label}")
                return summary

            cluster_summaries = await asyncio.gather(
                *[_gen_summary(c) for c in clusters]
            )
            cluster_summaries = list(cluster_summaries)

            # ③ 生成 fallback 报告（始终保存）
            all_years = []
            for data in cluster_data.values():
                all_years.extend([p.year for p in data["papers"] if p.year])
            total_papers_count = sum(len(data["papers"]) for data in cluster_data.values())
            fallback_report = self._fallback_report(
                rq_tree, cluster_summaries, all_years, total_papers_count
            )
            fallback_path = await self._save_report(fallback_report, suffix="_fallback")
            self.log_progress(f"Fallback report saved: {fallback_path}")

            # ④ 生成 LLM 最终报告（带重试）
            self.log_progress("Generating final literature review report...")
            final_report = None
            for attempt in range(3):
                if attempt > 0:
                    delay = 15.0 * attempt
                    self.log_progress(f"Retry {attempt + 1}/3 after {delay:.0f}s...")
                    await asyncio.sleep(delay)
                final_report = await self._generate_final_report(
                    rq_tree,
                    cluster_summaries,
                    cluster_data,
                    lang_constraint=lang_constraint,
                )
                if final_report:
                    break

            # 所有重试失败则使用 fallback 作为最终报告
            if not final_report:
                self.log_progress("All retries failed, using fallback report as final", "warning")
                final_report = fallback_report

            if not final_report:
                return self._create_result(
                    success=False,
                    errors=["Failed to generate final report (LLM rate limited)"]
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

        except KeyboardInterrupt:
            raise  # 重新抛出，让 BaseAgent.run() 处理
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
                    "abstract": (paper.abstract or "")[:500],
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
        lang_constraint: str = "",
    ) -> Dict[str, str]:
        """
        生成单个簇的摘要

        Args:
            cluster: 簇信息
            data: 簇数据
            include_methodology: 是否包含方法论分析
            include_applications: 是否包含应用分析
            lang_constraint: 语言约束指令

        Returns:
            摘要内容字典
        """
        content_parts = []

        content_parts.append(f"## {cluster.label}\n")
        content_parts.append(f"{cluster.description}\n")

        papers_summary_text = "\n".join([
            f"- [{p['year']}] {p['title']}\n  Authors: {', '.join(p['authors'])}\n  {p['abstract']}"
            for p in data["papers_summary"]
        ])

        if include_methodology:
            try:
                messages = [
                    SystemMessage(content="You are a senior academic literature review expert."),
                    HumanMessage(
                        content=self.METHODOLOGY_PROMPT.format(
                            lang_constraint=lang_constraint,
                            cluster_label=cluster.label,
                            cluster_description=cluster.description,
                            papers_summary=papers_summary_text,
                        )
                    ),
                ]

                response = await self._call_llm(messages)
                content_parts.append("### Methodology Analysis\n")
                content_parts.append(response)
                content_parts.append("\n")

            except Exception as e:
                self.log_progress(f"Methodology analysis failed for cluster {cluster.cluster_id}: {e}", "warning")

        if include_applications:
            try:
                messages = [
                    SystemMessage(content="You are a senior academic literature review expert."),
                    HumanMessage(
                        content=self.APPLICATION_PROMPT.format(
                            lang_constraint=lang_constraint,
                            cluster_label=cluster.label,
                            cluster_description=cluster.description,
                            papers_summary=papers_summary_text,
                        )
                    ),
                ]

                response = await self._call_llm(messages)
                content_parts.append("### Application Analysis\n")
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
        lang_constraint: str = "",
    ) -> str:
        """
        生成最终综述报告（拆分为四次LLM调用，避免单次输出被截断）

        每次调用只负责一个章节，大幅降低单次输出 token 量。

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

        # 估计年份范围
        all_years = []
        for data in cluster_data.values():
            all_years.extend([p.year for p in data["papers"] if p.year])
        year_range = (
            f"{min(all_years) if all_years else 'N/A'}-{max(all_years) if all_years else 'N/A'}"
        )

        total_papers = sum(len(data["papers"]) for data in cluster_data.values())
        topic = rq_tree.research_topic

        # ──── 公共写作规则 ────
        _WRITING_RULES = """【Writing Rules - ZERO TOLERANCE for violations】
- Every method MUST be accompanied by its core formula with EVERY symbol explained (meaning, dimension, typical range)
- "This method achieved good results" without specific metric values is FORBIDDEN
- Every technical conclusion must be traceable to a specific formula, equation, or physical law
- Do not start with cliches like "In recent years" or "With the development of XX"
- Do not reduce technical principles to a single sentence like "utilizes XX technology" — explain the FORMULA
- Do not fabricate citations. All citations must come from real papers listed in the cluster summaries above

【Citation Format - Must Strictly Follow】
Correct: Zhang et al. proposed a new method [Zhang, 2023, A Novel Method for XX]
Correct: This method was first proposed in [Li, 2021, Deep Learning Based Optimization]
Wrong: [Zhang, 2023] ← missing title, not allowed
Wrong: [1] ← missing author, year, and title, not allowed
Do NOT cite papers not listed in the cluster summaries above."""

        # ──────────────────────────────────────────────
        # Part 1: 引言（输入完整摘要作为上下文，输出只需引言）
        # ──────────────────────────────────────────────
        part1_prompt = f"""{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.

Research Topic: {topic}
Review Period: {year_range}
Total Papers Analyzed: {total_papers}
Number of Topic Clusters: {len(cluster_summaries)}

Cluster Analysis Summaries (for context only — do NOT repeat these summaries, write original prose):
{cluster_summaries_text}

{_WRITING_RULES}

Please write ONLY Section 1 of the literature review:

## 1. Introduction
### 1.1 Research Background and Core Challenges
(Point out the most fundamental physical/technical bottlenecks, explain WHY this problem is inherently difficult from a physics/math perspective — which fundamental law or equation creates the constraint?)
### 1.2 Scope of Review and Methodology
(Describe the systematic review methodology, data sources, number of papers analyzed, and clustering approach)

Write concisely — this section should be about 500-800 words. Do NOT write any other sections."""

        self.log_progress("Generating report part 1/4: Introduction...")
        messages_1 = [
            SystemMessage(content="You are a senior academic literature review expert."),
            HumanMessage(content=part1_prompt),
        ]
        part1 = await self._call_llm(messages_1)

        # ──────────────────────────────────────────────
        # Part 2: 核心技术原理（每个簇的方法论 + 公式）
        # ──────────────────────────────────────────────
        part2_prompt = f"""{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.
Your core principle: **Focus ONLY on underlying principles, core formulas, and mathematical derivations.**

Research Topic: {topic}
Number of Topic Clusters: {len(cluster_summaries)}

Cluster Analysis Summaries:
{cluster_summaries_text}

{_WRITING_RULES}

Please write ONLY Section 2 of the literature review. Write a subsection (### 2.x Title) for EACH of the {len(cluster_summaries)} topic clusters.

## 2. Core Technology Principles

### 2.x [Cluster Theme]
#### 2.x.1 Core Formulas and Derivation
- Write out the key equation(s) for methods in this cluster
- Explain EVERY symbol: its physical meaning, dimension, and typical value range
- Show the derivation chain: from which physical law or theorem is this formula derived?
- State the key assumptions and their validity range

#### 2.x.2 Key Parameters and Theoretical Limits
- Which parameter determines the performance ceiling? Why? (explain from the formula structure)
- Derive the theoretical limit from first principles
- Compare theoretical limit with reported experimental values

#### 2.x.3 Principle-Level Breakthrough
- What fundamental formula/physics change did this method make compared to prior methods?
- Which term in the equation was modified, added, or removed?
- What new constraint did this create?

#### 2.x.4 Quantitative Comparison
| Method | Core Formula | Key Parameters | Metric Value (Reported) | Theoretical Limit (Derived) | Gap |
|--------|-------------|----------------|------------------------|----------------------------|-----|

Note: You MUST cover ALL {len(cluster_summaries)} topic clusters. Do not write any other sections."""

        self.log_progress("Generating report part 2/4: Core Technology Principles...")
        messages_2 = [
            SystemMessage(content="You are a senior academic literature review expert."),
            HumanMessage(content=part2_prompt),
        ]
        part2 = await self._call_llm(messages_2)

        # ──────────────────────────────────────────────
        # Part 3: 应用场景 + 技术演进
        # ──────────────────────────────────────────────
        part3_prompt = f"""{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.
Core principle: **Focus on underlying formulas, physical mechanisms, and mathematical derivations.**

Research Topic: {topic}
Review Period: {year_range}
Total Papers Analyzed: {total_papers}

Cluster Analysis Summaries:
{cluster_summaries_text}

{_WRITING_RULES}

Please write ONLY Sections 3 and 4 of the literature review:

## 3. Technical Performance in Practice
Categorize by application scenario. Each scenario MUST include:
- The core formula adapted for this scenario (show how it changes from the base formula)
- Specific quantitative metric values (e.g., SS reduced from X mV/dec to Y mV/dec)
- Performance gap between actual and ideal: which formula term accounts for the gap?
- Formula-level explanation for WHY performance differs across scenarios

## 4. Technology Evolution and Physical Limits
### 4.1 Formula Evolution Across Generations
For each generational transition:
- Write out the core formula BEFORE and AFTER the change
- Highlight exactly which term was modified/added/removed
- Explain WHY this formula change was necessary (which prior limitation did it address?)
- What new constraint did the modified formula introduce?
- Show the quantitative impact of the formula change on performance

### 4.2 Theoretical Limit Analysis from First Principles
- Starting from fundamental physical laws, derive the theoretical performance upper/lower bound
- Write out the derivation chain: fundamental law → intermediate equations → practical limit
- Compare the theoretical limit with the best reported experimental value
- Identify which step introduces the largest gap (approximation loss)
- Is the remaining gap due to the physics (irreducible) or engineering (potentially improvable)?

### 4.3 Cross-domain Principle Transfer Feasibility
When importing methods from other domains:
- Compare the core formulas: are the mathematical structures compatible?
- List the specific conditions (parameter ranges, assumptions) for transfer validity
- What formula modifications are needed? What new terms must be added?

Do not write any other sections."""

        self.log_progress("Generating report part 3/4: Applications + Evolution...")
        messages_3 = [
            SystemMessage(content="You are a senior academic literature review expert."),
            HumanMessage(content=part3_prompt),
        ]
        part3 = await self._call_llm(messages_3)

        # ──────────────────────────────────────────────
        # Part 4: 根因分析 + 结论
        # ──────────────────────────────────────────────
        part4_prompt = f"""{lang_constraint}

You are a senior academic literature review expert with deep expertise in mathematical modeling and first-principles analysis.
Core principle: **Focus on underlying formulas, physical mechanisms, and mathematical derivations.**

Research Topic: {topic}

Cluster Analysis Summaries:
{cluster_summaries_text}

{_WRITING_RULES}

Please write ONLY Sections 5 and 6 of the literature review:

## 5. Root Cause Analysis and Breakthrough Paths
### 5.1 Formula-Level Root Causes of Core Bottlenecks
The common bottleneck faced by all current methods:
- Which specific term in which formula creates this bottleneck?
- Trace back to the fundamental physical law that imposes this constraint
- Is it possible to circumvent this term, or is it an irreducible consequence of physics?

### 5.2 Most Promising Breakthrough Directions
For each proposed direction:
- Show the formula modification it would require
- Prove feasibility from first principles (why the modified formula is mathematically/physically valid)
- Estimate the theoretical performance improvement if successful
- Identify prerequisites that must be met (in terms of parameter values or material properties)

### 5.3 Unsolved Fundamental Mathematical/Physical Problems
- List the open theoretical problems that, if solved, would unlock major performance gains
- For each problem, state which formula or physical law it relates to
- Explain why current mathematical/physical tools are insufficient

### 5.4 Recommendations for Researchers and Engineers
- For researchers: which formula to derive, which physical effect to model, which assumption to challenge
- For engineers: which parameter to optimize, which manufacturing tolerance is critical, and why (from the formula)

## 6. Conclusion
Must include:
- Core formula-level findings (the most important equations and what they tell us)
- Quantitative summary of how close current methods are to theoretical limits
- The single most impactful direction for future research, justified from first principles

Requirements:
- Avoid hollow statements like "this method has broad prospects" or "this field has great potential"
- Write as a technical report for domain experts, emphasizing formulas, derivations, and numerical values

Do not write any other sections."""

        self.log_progress("Generating report part 4/4: Root Causes + Conclusion...")
        messages_4 = [
            SystemMessage(content="You are a senior academic literature review expert."),
            HumanMessage(content=part4_prompt),
        ]
        part4 = await self._call_llm(messages_4)

        # 组合完整报告
        return f"{part1}\n\n{part2}\n\n{part3}\n\n{part4}"

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

    async def _save_report(self, report: str, suffix: str = "") -> str:
        """
        保存报告

        Args:
            report: 报告内容
            suffix: 文件名后缀（如 "_fallback"）

        Returns:
            保存路径
        """
        reports_dir = Path(self.workspace.base_path) / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = reports_dir / f"literature_review_{timestamp}{suffix}.md"

        async with aiofiles.open(str(report_path), "w", encoding="utf-8") as f:
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
