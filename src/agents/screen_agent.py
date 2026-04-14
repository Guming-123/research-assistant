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

from ..core.agent import BaseAgent, AgentConfig, AgentResult
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

请从以下维度评估：
1. 主题相关性（1-5分）：论文是否直接回答RQ？
2. 方法相关性（1-5分）：论文方法是否属于RQ关注范围？
3. 时效性（1-5分）：论文是否反映最新进展？

输出JSON格式：
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "relevance_scores": {{
    "topic": 分数,
    "method": 分数,
    "timeliness": 分数
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
            model="glm-4-plus",
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
            threshold: NF阈值（默认0.7）
            use_llm: 是否使用LLM辅助判定（默认True）
            llm_threshold: LLM判定的NF阈值范围（默认0.5-0.8）

        Returns:
            AgentResult
        """
        rq_ids = kwargs.get("rq_ids", [])
        threshold = kwargs.get("threshold", 0.7)
        use_llm = kwargs.get("use_llm", True)
        llm_threshold = kwargs.get("llm_threshold", (0.5, 0.8))

        try:
            # 获取文献池
            all_papers = await self.workspace.get_literature()
            if not all_papers:
                return self._create_result(
                    success=False,
                    errors=["No papers in workspace"]
                )

            self.log_progress(f"Screening {len(all_papers)} papers...")

            # 获取RQ信息
            rq_questions = []
            for rq_id in rq_ids:
                rq = self.rq_manager.get_question(rq_id)
                if rq:
                    rq_questions.append(f"{rq.id}: {rq.question}")

            # ① 文档解析与分块
            self.log_progress("Parsing and chunking documents...")
            self._paper_chunks = await self._parse_and_chunk(all_papers)

            # 保存chunk计数信息（用于NF计算）
            paper_chunk_counts = {pid: len(chunks) for pid, chunks in self._paper_chunks.items()}
            await self._save_to_workspace("paper_chunk_counts", paper_chunk_counts, stage="screen")

            # ② 向量化与索引构建
            self.log_progress("Building embeddings and FAISS index...")
            await self._build_vector_index(self._paper_chunks)

            # ③ 基于RQ的语义检索
            self.log_progress("Performing semantic retrieval based on RQs...")
            retrieval_results = await self._semantic_retrieval(rq_ids)

            # ④ 归一化频率过滤（传入实际的chunk计数）
            self.log_progress(f"Applying normalized frequency filter (threshold={threshold})...")
            nf_results = self._apply_nf_filter(
                retrieval_results,
                paper_chunk_counts,
                threshold
            )

            # ⑤ LLM辅助相关性判定（可选）
            screening_results = []
            if use_llm:
                self.log_progress("Running LLM-assisted relevance judgment...")
                screening_results = await self._llm_screening(
                    nf_results,
                    rq_questions,
                    llm_threshold
                )
            else:
                # 只使用NF结果
                for paper_id, nf_score in nf_results.items():
                    screening_results.append(ScreeningResult(
                        paper_id=paper_id,
                        relevant=True,
                        confidence=min(nf_score, 1.0),
                        relevance_scores={"nf": nf_score},
                        reasoning=f"Normalized frequency: {nf_score:.2f}",
                        normalized_frequency=nf_score,
                        related_rqs=[],
                    ))

            # 更新文献记录
            relevant_count = 0
            for result in screening_results:
                await self.workspace.update_literature(
                    result.paper_id,
                    {"relevance_score": result.confidence}
                )
                if result.relevant:
                    relevant_count += 1

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
                "avg_chunks_per_paper": sum(paper_chunk_counts.values()) / len(paper_chunk_counts) if paper_chunk_counts else 0,
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

        # 保存chunk embeddings（使用chunk_id作为key，避免覆盖）
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            await self.workspace.save(chunk_id, embedding, agent=self.name, stage="screen")

        # 为每篇论文聚合所有chunk的embeddings
        paper_embeddings = {}
        for chunk_id, embedding in zip(chunk_ids, embeddings):
            paper_id = chunk_to_paper[chunk_id]
            if paper_id not in paper_embeddings:
                paper_embeddings[paper_id] = []
            paper_embeddings[paper_id].append(embedding)

        # 计算每篇论文的平均embedding并保存
        for paper_id, emb_list in paper_embeddings.items():
            avg_embedding = list(np.mean(emb_list, axis=0))
            await self.workspace.save_embedding(paper_id, avg_embedding)

        # 构建FAISS索引
        self.faiss_index = await build_faiss_index(embeddings, chunk_ids)

        self.log_progress(f"Built index with {len(embeddings)} chunks from {len(paper_chunks)} papers")

    async def _semantic_retrieval(
        self,
        rq_ids: List[str],
        top_k: int = 10,
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

    async def _llm_screening(
        self,
        nf_results: Dict[str, float],
        rq_questions: List[str],
        llm_threshold: Tuple[float, float] = (0.5, 0.8),
    ) -> List[ScreeningResult]:
        """
        LLM辅助相关性判定

        优化：只对边界案例调用LLM

        Args:
            nf_results: NF过滤后的论文
            rq_questions: RQ问题列表
            llm_threshold: (min, max) NF范围，在此范围内才调用LLM

        Returns:
            筛选结果列表
        """
        results = []
        llm_min, llm_max = llm_threshold

        for paper_id, nf_score in nf_results.items():
            try:
                # 高置信度：双模型精筛（如果配置了refinement_llm）
                if nf_score >= llm_max:
                    if self.refinement_llm:
                        # 先用screening_llm快速验证
                        screen_result = await self._quick_screen(paper_id, rq_questions)
                        if screen_result is None or screen_result.relevant:
                            # 通过快速验证，使用refinement_llm进行精筛
                            refined_result = await self._refine_screening(paper_id, rq_questions, nf_score)
                            results.append(refined_result)
                        else:
                            # 快速验证未通过
                            results.append(screen_result)
                    else:
                        # 没有refinement_llm，直接通过
                        results.append(ScreeningResult(
                            paper_id=paper_id,
                            relevant=True,
                            confidence=min(nf_score, 1.0),
                            relevance_scores={"nf": nf_score},
                            reasoning=f"High normalized frequency: {nf_score:.2f} (auto-approved)",
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
                        relevance_scores={"nf": nf_score},
                        reasoning=f"Low normalized frequency: {nf_score:.2f}",
                        normalized_frequency=nf_score,
                        related_rqs=[],
                    ))
                    continue

                # 边界案例：调用LLM
                papers = await self.workspace.get_literature(paper_ids=[paper_id])
                if not papers:
                    continue

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

                # 使用筛选模型（低成本）进行LLM调用
                try:
                    llm_response = await self.screening_llm.ainvoke(messages)
                    llm_result = json.loads(llm_response.content)
                except (json.JSONDecodeError, Exception) as e:
                    # Fallback: 使用基类方法
                    self.log_progress(f"Screening LLM failed, using fallback: {e}", "warning")
                    response = await self._call_llm(messages, response_format="json")
                    llm_result = json.loads(response)

                result = ScreeningResult(
                    paper_id=paper_id,
                    relevant=llm_result.get("relevant", False),
                    confidence=llm_result.get("confidence", 0.5),
                    relevance_scores=llm_result.get("relevance_scores", {}),
                    reasoning=llm_result.get("reasoning", ""),
                    normalized_frequency=nf_score,
                    related_rqs=llm_result.get("related_rqs", []),
                )

                results.append(result)

            except Exception as e:
                self.log_progress(f"LLM screening failed for {paper_id}: {e}", "warning")
                # Fallback: 使用NF结果
                results.append(ScreeningResult(
                    paper_id=paper_id,
                    relevant=nf_score >= threshold,
                    confidence=min(nf_score, 1.0),
                    relevance_scores={"nf": nf_score},
                    reasoning=f"NF score (LLM failed): {nf_score:.2f}",
                    normalized_frequency=nf_score,
                    related_rqs=[],
                ))

        return results
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
            response = await self.screening_llm.ainvoke(messages)
            llm_result = json.loads(response.content)

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
            response = await self.refinement_llm.ainvoke(messages)
            llm_result = json.loads(response.content)

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
