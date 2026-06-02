"""
Screen Agent - 相关性筛选Agent
负责对文献池进行相关性筛选，替代人工初筛
"""

import asyncio
import json
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import logging
import numpy as np

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..core.agent import BaseAgent, AgentConfig, AgentResult, get_agent_model
from ..core.workspace import LiteratureRecord, SharedWorkspace
from ..utils.text import chunk_text
from ..utils.embedding import get_embeddings, build_faiss_index
from ..utils.llm import get_llm_client
from ..utils.exceptions import ScreeningError, LLMError, ValidationError
from ..config import ScreenConfig

logger = logging.getLogger(__name__)


@dataclass
class ScreeningResult:
    """筛选结果"""
    paper_id: str
    relevant: bool
    confidence: float
    relevance_scores: Dict[str, float]
    reasoning: str
    normalized_frequency: float
    related_rqs: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "relevant": self.relevant,
            "confidence": self.confidence,
            "relevance_scores": self.relevance_scores,
            "reasoning": self.reasoning,
            "normalized_frequency": self.normalized_frequency,
            "related_rqs": self.related_rqs,
        }


class ScreenAgent(BaseAgent):
    """
    相关性筛选Agent

    职责：
    ① 文档解析与分块
    ② 向量化与索引构建
    ③ 基于RQ的语义检索
    ④ 归一化频率过滤
    ⑤ LLM辅助相关性判定
    """

    # LLM相关性判定Prompt
    LLM_SCREENING_PROMPT = """你是一个文献筛选专家。请判断以下论文是否与研究问题相关。

研究问题（RQ）：
{research_questions}

论文信息：
标题：{title}
摘要：{abstract}

请从以下维度评估（每项1-5分）：
1. 主题相关性（topic）：论文是否直接回答RQ？
2. 方法相关性（method）：论文方法是否属于RQ关注范围？
3. 时效性（timeliness）：论文是否反映最新进展？

判定规则：relevant = (topic * 0.5 + method * 0.3 + timeliness * 0.2) >= 3.0
必须严格按此公式计算，relevant 为 true 当且仅当加权总分 >= 3.0。

输出JSON格式：
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "relevance_scores": {{
    "topic": 分数,
    "method": 分数,
    "timeliness": 分数,
    "weighted_total": 加权总分
  }},
  "reasoning": "判断理由（一句话）",
  "related_rqs": ["RQ1", "RQ2", ...]
}}"""

    def __init__(
        self,
        workspace: SharedWorkspace,
        rq_manager,
        llm_client: Optional[ChatOpenAI] = None,
        config: Optional[ScreenConfig] = None,
    ):
        """
        初始化Screen Agent

        Args:
            workspace: 共享工作区
            rq_manager: RQ管理器
            llm_client: LLM客户端
            config: Agent配置
        """
        config = config or ScreenConfig(
            name="ScreenAgent",
            description="Screens literature for relevance to research questions",
            model=get_agent_model("screen"),
            temperature=0.3,
        )
        super().__init__(config, workspace, llm_client)

        self.rq_manager = rq_manager
        self.faiss_index = None
        self._paper_chunks: Dict[str, List[Dict[str, Any]]] = {}  # 存储chunk信息

        # 模型分层配置：创建筛选模型的LLM客户端
        self.screening_llm = get_llm_client(
            model=config.screening_model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

        # 创建精确判定模型的LLM客户端（如果配置了的话）
        if config.refinement_model:
            self.refinement_llm = get_llm_client(
                model=config.refinement_model,
                temperature=config.temperature * 0.8,  # 稍低的温度以获得更一致的判定
                max_tokens=config.max_tokens,
            )
        else:
            self.refinement_llm = None

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        return "rq_ids" in kwargs

    async def execute(self, **kwargs) -> AgentResult:
        """
        执行相关性筛选

        Args:
            rq_ids: 相关的RQ ID列表
            threshold: 相似度阈值（默认0.3）
            use_llm: 是否使用LLM辅助判定（默认True）
            llm_threshold: LLM判定的相似度阈值范围（默认0.35-0.55）

        Returns:
            AgentResult
        """
        rq_ids = kwargs.get("rq_ids", [])
        threshold = kwargs.get("threshold", 0.3)
        use_llm = kwargs.get("use_llm", True)
        llm_threshold = kwargs.get("llm_threshold", (0.61, 0.68))

        try:
            # 获取文献池：优先只筛选当前搜索的论文，避免旧主题论文污染
            search_ids = await self.workspace.get_metadata_item("current_search_paper_ids")
            if search_ids and isinstance(search_ids, list):
                all_papers = await self.workspace.get_literature(paper_ids=search_ids)
                self.log_progress(
                    f"Screening {len(all_papers)} papers from current search "
                    f"(excluded {len(search_ids) - len(all_papers)} missing)"
                )
            else:
                # 回退：无搜索记录时加载全部，先重置所有分数防止旧数据污染
                self.log_progress(
                    "No search session found, screening all papers with score reset",
                    "warning",
                )
                await self.workspace.reset_all_relevance_scores()
                all_papers = await self.workspace.get_literature()

            if not all_papers:
                return self._create_result(
                    success=False,
                    errors=["No papers in workspace"]
                )

            # 获取RQ信息
            rq_questions = []
            for rq_id in rq_ids:
                rq = self.rq_manager.get_question(rq_id)
                if rq:
                    rq_questions.append(f"{rq.id}: {rq.question}")

            # ① 论文级相关性评分（全覆盖，不依赖chunk检索）
            self.log_progress("Computing paper-level relevance scores...")
            relevance_scores = await self._compute_paper_relevance(all_papers, rq_ids)

            # ② 过滤低相关性论文
            filtered = {
                pid: score for pid, score in relevance_scores.items()
                if score >= threshold
            }
            self.log_progress(
                f"Relevance filter: {len(relevance_scores)} -> {len(filtered)} papers "
                f"(threshold={threshold})"
            )

            # ③ LLM辅助判定
            screening_results = []
            if use_llm:
                self.log_progress("Running LLM-assisted relevance judgment...")
                screening_results = await self._llm_screening(
                    filtered,
                    rq_questions,
                    llm_threshold,
                    fallback_threshold=threshold,
                )
            else:
                for paper_id, score in filtered.items():
                    screening_results.append(ScreeningResult(
                        paper_id=paper_id,
                        relevant=True,
                        confidence=min(score, 1.0),
                        relevance_scores={"similarity": score},
                        reasoning=f"Similarity score: {score:.3f}",
                        normalized_frequency=score,
                        related_rqs=[],
                    ))

            # 批量更新论文相关度分数（仅当前搜索的论文，不触碰旧主题数据）
            relevance_updates = {p.id: None for p in all_papers}
            relevant_count = 0
            for result in screening_results:
                if result.relevant:
                    relevance_updates[result.paper_id] = result.confidence
                    relevant_count += 1
            await self.workspace.batch_update_relevance(relevance_updates)

            # 保存筛选结果
            await self._save_to_workspace(
                "screening_results",
                [r.to_dict() for r in screening_results],
                stage="screen"
            )

            metrics = {
                "total_papers": len(all_papers),
                "relevant_papers": relevant_count,
                "relevance_rate": relevant_count / len(all_papers) if all_papers else 0,
                "threshold": threshold,
                "llm_assisted": use_llm,
                "scored_papers": len(relevance_scores),
                "filtered_papers": len(filtered),
            }

            self.log_progress(
                f"Screening complete: {relevant_count}/{len(all_papers)} papers "
                f"({metrics['relevance_rate']:.1%}) marked as relevant"
            )

            return self._create_result(
                success=True,
                data={"relevant_count": relevant_count},
                metrics=metrics,
            )

        except KeyboardInterrupt:
            raise  # 重新抛出，让 BaseAgent.run() 处理
        except (LLMError, ValidationError) as e:
            error_msg = f"Screening failed: {type(e).__name__}: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])
        except (KeyError, ValueError) as e:
            error_msg = f"Data processing error: {type(e).__name__}: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])
        except Exception as e:
            error_msg = f"Unexpected screening error: {type(e).__name__}: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

    async def _parse_and_chunk(
        self,
        papers: List[LiteratureRecord],
        chunk_size: int = 512,
        chunk_overlap: int = 50,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        解析和分块文档

        Args:
            papers: 文献列表
            chunk_size: 块大小
            chunk_overlap: 块重叠

        Returns:
            {paper_id: [chunk1, chunk2, ...]}
        """
        paper_chunks = {}

        for paper in papers:
            try:
                # 组合文本（标题 + 摘要）
                text = paper.full_text or f"{paper.title}\n\n{paper.abstract}"

                if not text or len(text.strip()) < 50:
                    continue

                # 分块
                chunks = await chunk_text(text, chunk_size, chunk_overlap)

                paper_chunks[paper.id] = [
                    {
                        "text": chunk,
                        "paper_id": paper.id,
                        "index": i,
                        "char_count": len(chunk)
                    }
                    for i, chunk in enumerate(chunks)
                ]

            except Exception as e:
                self.log_progress(f"Failed to chunk paper {paper.id}: {e}", "warning")
                continue

        self.log_progress(f"Chunked {len(paper_chunks)} papers")
        return paper_chunks

    async def _build_vector_index(
        self,
        paper_chunks: Dict[str, List[Dict[str, Any]]],
    ) -> None:
        """
        构建向量索引

        修复：使用chunk_id存储embedding，避免覆盖

        Args:
            paper_chunks: 论文分块
        """
        # 收集所有chunks
        all_chunks = []
        chunk_ids = []
        chunk_to_paper = {}  # 记录chunk_id到paper_id的映射

        for paper_id, chunks in paper_chunks.items():
            for chunk in chunks:
                # 使用安全的分隔符（避免paper_id中包含下划线导致解析错误）
                chunk_id = f"{paper_id}::{chunk['index']}"
                all_chunks.append(chunk["text"])
                chunk_ids.append(chunk_id)
                chunk_to_paper[chunk_id] = paper_id

        if not all_chunks:
            self.log_progress("No chunks to index", "warning")
            return

        # 生成embeddings
        embeddings = await get_embeddings(all_chunks)

        # 批量保存chunk embeddings到workspace
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            await self.workspace.save_metadata_item(
                chunk_id, embedding, agent=self.name, stage="screen"
            )

        # 为每篇论文聚合所有chunk的embeddings
        paper_embeddings = {}
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            paper_id = chunk_to_paper[chunk_id]
            if paper_id not in paper_embeddings:
                paper_embeddings[paper_id] = []
            paper_embeddings[paper_id].append(embedding)

        # 批量计算平均embedding并保存
        avg_embeddings = {
            pid: list(np.mean(embs, axis=0))
            for pid, embs in paper_embeddings.items()
        }
        await self.workspace.batch_save_embeddings(avg_embeddings)

        # 构建FAISS索引
        self.faiss_index = await build_faiss_index(embeddings, chunk_ids)

        self.log_progress(f"Built index with {len(embeddings)} chunks from {len(paper_chunks)} papers")

    async def _semantic_retrieval(
        self,
        rq_ids: List[str],
        top_k: int = 50,
    ) -> Dict[str, Set[int]]:
        """
        基于RQ进行语义检索

        Args:
            rq_ids: RQ ID列表
            top_k: 每个RQ检索的top-k chunks

        Returns:
            {paper_id: set(retrieved_chunk_indices)}
        """
        if not self.faiss_index:
            return {}

        paper_retrievals: Dict[str, Set[int]] = {}

        for rq_id in rq_ids:
            rq = self.rq_manager.get_question(rq_id)
            if not rq:
                continue

            # 使用RQ问题作为查询
            query_text = f"{rq.question} {' '.join(rq.keywords)}"
            query_embedding = await get_embeddings([query_text])

            # 检索
            results = self.faiss_index.search(query_embedding[0], top_k=top_k)

            # 记录检索到的chunks
            for chunk_id, score in results:
                if score > 0.5:  # 相似度阈值
                    # 使用安全的分隔符解析
                    paper_id, chunk_index_str = chunk_id.rsplit("::", 1)
                    chunk_index = int(chunk_index_str)

                    if paper_id not in paper_retrievals:
                        paper_retrievals[paper_id] = set()
                    paper_retrievals[paper_id].add(chunk_index)

        return paper_retrievals

    def _apply_nf_filter(
        self,
        retrieval_results: Dict[str, Set[int]],
        paper_chunk_counts: Dict[str, int],
        threshold: float,
    ) -> Dict[str, float]:
        """
        应用归一化频率过滤

        修复：使用实际的chunk计数而非硬编码

        Args:
            retrieval_results: 检索结果 {paper_id: set(retrieved_chunk_indices)}
            paper_chunk_counts: 每篇论文的实际chunk数 {paper_id: count}
            threshold: NF阈值

        Returns:
            {paper_id: nf_score}
        """
        nf_scores = {}

        for paper_id, retrieved_chunks in retrieval_results.items():
            # 获取实际chunk数
            total_chunks = paper_chunk_counts.get(paper_id, 1)  # 至少为1避免除零
            nf = len(retrieved_chunks) / total_chunks
            nf_scores[paper_id] = nf

        # 过滤
        filtered = {
            pid: nf
            for pid, nf in nf_scores.items()
            if nf >= threshold
        }

        self.log_progress(
            f"NF filter: {len(nf_scores)} candidates -> {len(filtered)} papers "
            f"(threshold={threshold})"
        )

        return filtered

    async def _compute_paper_relevance(
        self,
        papers: List[LiteratureRecord],
        rq_ids: List[str],
    ) -> Dict[str, float]:
        """
        论文级相关性评分：直接计算每篇论文与每个RQ的余弦相似度

        替代旧的 chunk-based NF 方法。优势：
        1. 覆盖全部论文（不受 top_k 限制）
        2. 论文级匹配更稳定（不受 chunk 分割影响）
        3. 计算效率高（一次矩阵乘法）

        Args:
            papers: 论文列表
            rq_ids: RQ ID列表

        Returns:
            {paper_id: max_cosine_similarity_to_any_RQ}
        """
        paper_texts = []
        paper_ids = []
        for p in papers:
            text = f"{p.title}\n{p.abstract or ''}"
            if len(text.strip()) >= 50:
                paper_texts.append(text)
                paper_ids.append(p.id)

        if not paper_texts or not rq_ids:
            return {}

        # 生成论文 embeddings
        self.log_progress(f"Computing embeddings for {len(paper_texts)} papers...")
        paper_embeddings = await get_embeddings(paper_texts)

        # 生成 RQ embeddings
        rq_texts = []
        for rq_id in rq_ids:
            rq = self.rq_manager.get_question(rq_id)
            if rq:
                rq_texts.append(f"{rq.question} {' '.join(rq.keywords)}")

        rq_embeddings = await get_embeddings(rq_texts)

        # 余弦相似度矩阵 (papers × RQs)
        paper_arr = np.array(paper_embeddings, dtype=np.float32)
        rq_arr = np.array(rq_embeddings, dtype=np.float32)

        # L2 归一化
        p_norms = np.linalg.norm(paper_arr, axis=1, keepdims=True)
        p_norms[p_norms == 0] = 1
        paper_arr /= p_norms

        r_norms = np.linalg.norm(rq_arr, axis=1, keepdims=True)
        r_norms[r_norms == 0] = 1
        rq_arr /= r_norms

        # 每篇论文取与所有 RQ 的最大相似度
        sim_matrix = paper_arr @ rq_arr.T
        max_scores = sim_matrix.max(axis=1)

        self.log_progress(
            f"Score distribution: min={max_scores.min():.3f}, "
            f"p25={np.percentile(max_scores, 25):.3f}, "
            f"median={np.median(max_scores):.3f}, "
            f"p75={np.percentile(max_scores, 75):.3f}, "
            f"max={max_scores.max():.3f}"
        )

        return {pid: float(score) for pid, score in zip(paper_ids, max_scores)}

    async def _llm_screening(
        self,
        nf_results: Dict[str, float],
        rq_questions: List[str],
        llm_threshold: Tuple[float, float] = (0.61, 0.68),
        fallback_threshold: float = 0.61,
    ) -> List[ScreeningResult]:
        """
        LLM辅助相关性判定（并行优化）

        分类策略：
        - 高相似度 (>= llm_max): 直接自动通过，不调用LLM
        - 低相似度 (< llm_min): 直接拒绝
        - 边界区域 (llm_min ~ llm_max): 按分数排序，取 top N 送LLM判定

        Args:
            nf_results: 相似度过滤后的论文 {paper_id: similarity_score}
            rq_questions: RQ问题列表
            llm_threshold: (min, max) 相似度范围
            fallback_threshold: LLM失败时的兜底阈值

        Returns:
            筛选结果列表
        """
        results: List[ScreeningResult] = []
        llm_min, llm_max = llm_threshold

        # 收集边界论文（按分数排序，限制数量）
        boundary_papers: List[Tuple[str, float]] = []
        max_llm_calls = 150

        for paper_id, nf_score in nf_results.items():
            # 高置信度：直接通过，不调用 LLM
            if nf_score >= llm_max:
                results.append(ScreeningResult(
                    paper_id=paper_id,
                    relevant=True,
                    confidence=min(nf_score, 1.0),
                    relevance_scores={"similarity": nf_score},
                    reasoning=f"High similarity: {nf_score:.3f} (auto-approved)",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                ))
                continue

            # 低置信度：直接拒绝
            if nf_score < llm_min:
                results.append(ScreeningResult(
                    paper_id=paper_id,
                    relevant=False,
                    confidence=nf_score,
                    relevance_scores={"similarity": nf_score},
                    reasoning=f"Low similarity: {nf_score:.3f}",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                ))
                continue

            # 边界案例：收集起来
            boundary_papers.append((paper_id, nf_score))

        # 边界论文按分数降序排列，只取 top N 送 LLM
        boundary_papers.sort(key=lambda x: x[1], reverse=True)
        if len(boundary_papers) > max_llm_calls:
            self.log_progress(
                f"Boundary zone has {len(boundary_papers)} papers, "
                f"sending top {max_llm_calls} to LLM"
            )
            # 超出限额的边界论文直接拒绝
            for paper_id, nf_score in boundary_papers[max_llm_calls:]:
                results.append(ScreeningResult(
                    paper_id=paper_id,
                    relevant=False,
                    confidence=nf_score,
                    relevance_scores={"similarity": nf_score},
                    reasoning=f"Boundary overflow: {nf_score:.3f}",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                ))
            boundary_papers = boundary_papers[:max_llm_calls]

        # 并行处理边界论文
        if boundary_papers:
            self.log_progress(f"LLM screening {len(boundary_papers)} boundary papers...")
            llm_tasks = [
                self._screen_boundary(pid, rq_questions, score, fallback_threshold)
                for pid, score in boundary_papers
            ]
            llm_results = await asyncio.gather(*llm_tasks, return_exceptions=True)
            for r in llm_results:
                if isinstance(r, ScreeningResult):
                    results.append(r)
                elif isinstance(r, Exception):
                    self.log_progress(f"LLM screening task failed: {r}", "warning")

        # 统计
        auto_approved = sum(1 for r in results if r.relevant and "auto" in r.reasoning)
        auto_rejected = sum(1 for r in results if not r.relevant and "auto" not in r.reasoning
                           and "overflow" not in r.reasoning)
        llm_screened = sum(1 for r in results if "auto" not in r.reasoning and "overflow" not in r.reasoning)
        self.log_progress(
            f"Screening breakdown: {auto_approved} auto-approved, "
            f"{llm_screened} LLM-screened, {auto_rejected} rejected"
        )

        return results

    async def _screen_high_confidence(
        self,
        paper_id: str,
        rq_questions: List[str],
        nf_score: float,
    ) -> ScreeningResult:
        """高置信度论文的精筛"""
        try:
            screen_result = await self._quick_screen(paper_id, rq_questions)
            if screen_result is None or screen_result.relevant:
                return await self._refine_screening(paper_id, rq_questions, nf_score)
            return screen_result
        except Exception as e:
            self.log_progress(f"High-confidence screening failed for {paper_id}: {e}", "warning")
            return ScreeningResult(
                paper_id=paper_id,
                relevant=True,
                confidence=min(nf_score, 1.0),
                relevance_scores={"nf": nf_score},
                reasoning=f"High NF, screening failed: {nf_score:.2f}",
                normalized_frequency=nf_score,
                related_rqs=[],
            )

    async def _screen_boundary(
        self,
        paper_id: str,
        rq_questions: List[str],
        nf_score: float,
        threshold: float,
    ) -> ScreeningResult:
        """边界案例的 LLM 筛选"""
        try:
            papers = await self.workspace.get_literature(paper_ids=[paper_id])
            if not papers:
                return ScreeningResult(
                    paper_id=paper_id,
                    relevant=False,
                    confidence=nf_score,
                    relevance_scores={"nf": nf_score},
                    reasoning="Paper not found",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                )

            paper = papers[0]
            messages = [
                SystemMessage(content="You are an expert literature reviewer."),
                HumanMessage(
                    content=self.LLM_SCREENING_PROMPT.format(
                        research_questions="\n".join(rq_questions),
                        title=paper.title,
                        abstract=paper.abstract or "No abstract available",
                    )
                ),
            ]

            try:
                response = await self._call_llm(messages, response_format="json")
                llm_result = json.loads(response)
            except Exception as e:
                self.log_progress(f"Screening LLM failed for {paper_id}: {e}", "warning")
                return ScreeningResult(
                    paper_id=paper_id,
                    relevant=nf_score >= threshold,
                    confidence=min(nf_score, 1.0),
                    relevance_scores={"similarity": nf_score},
                    reasoning=f"LLM failed, fallback to similarity: {nf_score:.3f}",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                )

            # 用公式校验 LLM 的 relevant 判定
            scores = llm_result.get("relevance_scores", {})
            weighted = (
                scores.get("topic", 3) * 0.5
                + scores.get("method", 3) * 0.3
                + scores.get("timeliness", 3) * 0.2
            )
            scores["weighted_total"] = round(weighted, 2)
            relevant = weighted >= 3.0

            return ScreeningResult(
                paper_id=paper_id,
                relevant=relevant,
                confidence=llm_result.get("confidence", 0.5),
                relevance_scores=scores,
                reasoning=llm_result.get("reasoning", ""),
                normalized_frequency=nf_score,
                related_rqs=llm_result.get("related_rqs", []),
            )

        except Exception as e:
            self.log_progress(f"LLM screening failed for {paper_id}: {e}", "warning")
            return ScreeningResult(
                paper_id=paper_id,
                relevant=nf_score >= threshold,
                confidence=min(nf_score, 1.0),
                relevance_scores={"nf": nf_score},
                reasoning=f"NF score (LLM failed): {nf_score:.2f}",
                normalized_frequency=nf_score,
                related_rqs=[],
            )
    async def _quick_screen(
        self,
        paper_id: str,
        rq_questions: List[str],
    ) -> Optional[ScreeningResult]:
        """
        快速筛选（使用screening_llm）

        Args:
            paper_id: 论文ID
            rq_questions: RQ问题列表

        Returns:
            筛选结果，如果快速验证通过返回None，未通过返回拒绝结果
        """
        papers = await self.workspace.get_literature(paper_ids=[paper_id])
        if not papers:
            return None

        paper = papers[0]
        messages = [
            SystemMessage(content="You are an expert literature reviewer."),
            HumanMessage(
                content=self.LLM_SCREENING_PROMPT.format(
                    research_questions="\n".join(rq_questions),
                    title=paper.title,
                    abstract=paper.abstract or "No abstract available",
                )
            ),
        ]

        try:
            response = await self._call_llm(messages, response_format="json")
            llm_result = json.loads(response)

            # 快速验证：如果相关性评分很低，返回拒绝结果
            relevance = llm_result.get("relevance_scores", {})
            topic_score = relevance.get("topic", 3)

            if topic_score < 3:  # 主题相关性低于3，认为快速验证未通过
                return ScreeningResult(
                    paper_id=paper_id,
                    relevant=False,
                    confidence=llm_result.get("confidence", 0.3),
                    relevance_scores=llm_result.get("relevance_scores", {}),
                    reasoning=f"Quick screening failed (topic score: {topic_score})",
                    normalized_frequency=0.0,
                    related_rqs=[],
                )
            return None  # 通过快速验证

        except Exception as e:
            self.log_progress(f"Quick screen failed for {paper_id}: {e}", "warning")
            return None  # 失败时默认通过，继续精筛

    async def _refine_screening(
        self,
        paper_id: str,
        rq_questions: List[str],
        nf_score: float = 0.0,
    ) -> ScreeningResult:
        """
        精确筛选（使用refinement_llm）

        Args:
            paper_id: 论文ID
            rq_questions: RQ问题列表
            nf_score: NF分数

        Returns:
            筛选结果
        """
        papers = await self.workspace.get_literature(paper_ids=[paper_id])
        if not papers:
            return ScreeningResult(
                paper_id=paper_id,
                relevant=False,
                confidence=0.0,
                relevance_scores={},
                reasoning="Paper not found",
                normalized_frequency=nf_score,
                related_rqs=[],
            )

        paper = papers[0]

        # 精筛prompt（更详细）
        refinement_prompt = """你是一个资深的文献评审专家。请对以下论文进行精确的相关性判定。

研究问题：
{research_questions}

论文详细信息：
标题：{title}
作者：{authors}
年份：{year}
摘要：{abstract}
来源：{venue}

请进行严格但公正的评估：
1. 该论文是否直接回答了研究问题？
2. 研究方法是否具有创新性？
3. 结果是否具有可信度？
4. 对该领域是否有重要贡献？

输出JSON格式：
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "relevance_scores": {{
    "topic": 分数(1-5),
    "method": 分数(1-5),
    "innovation": 分数(1-5),
    "credibility": 分数(1-5)
  }},
  "reasoning": "详细的判定理由（2-3句话）",
  "related_rqs": ["RQ1", "RQ2", ...]
}}"""

        messages = [
            SystemMessage(content="You are a senior literature reviewer."),
            HumanMessage(
                content=refinement_prompt.format(
                    research_questions="\n".join(rq_questions),
                    title=paper.title,
                    authors=", ".join(paper.authors[:5]) if paper.authors else "Unknown",
                    year=paper.year,
                    abstract=paper.abstract or "No abstract available",
                    venue=paper.venue or "Unknown venue",
                )
            ),
        ]

        try:
            response = await self._call_llm(messages, response_format="json")
            llm_result = json.loads(response)

            return ScreeningResult(
                paper_id=paper_id,
                relevant=llm_result.get("relevant", True),
                confidence=llm_result.get("confidence", 0.8),
                relevance_scores=llm_result.get("relevance_scores", {}),
                reasoning=llm_result.get("reasoning", "") + " [Refined]",
                normalized_frequency=nf_score,
                related_rqs=llm_result.get("related_rqs", []),
            )

        except Exception as e:
            self.log_progress(f"Refinement failed for {paper_id}: {e}", "warning")
            # 失败时返回通过结果（因为已经通过了快速筛选）
            return ScreeningResult(
                paper_id=paper_id,
                relevant=True,
                confidence=0.7,
                relevance_scores={},
                reasoning=f"Passed quick screen, refinement failed: {str(e)[:50]}",
                normalized_frequency=nf_score,
                related_rqs=[],
            )
