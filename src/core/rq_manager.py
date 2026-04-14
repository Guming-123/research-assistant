"""
Research Question Manager - 层级研究问题管理器
基于论文的层级RQ设计，用于驱动整个文献综述流程
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set
from enum import Enum
import json


class RQLevel(Enum):
    """研究问题层级"""
    LEVEL_1 = 1  # 宏观维度（方法论、应用、趋势等）
    LEVEL_2 = 2  # 中观分析（具体方法类型、具体领域）
    LEVEL_3 = 3  # 微观深入（技术细节、具体优缺点）


@dataclass
class ResearchQuestion:
    """
    研究问题（RQ）数据结构

    支持层级结构，每个RQ可以有多个子RQ
    """

    id: str  # 唯一标识，如 "RQ1", "RQ11", "RQ111"
    question: str  # 问题文本
    level: RQLevel  # 层级
    description: Optional[str] = None  # 详细描述
    parent_id: Optional[str] = None  # 父RQ ID
    children: List["ResearchQuestion"] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)  # 用于检索的关键词
    status: str = "pending"  # pending, in_progress, completed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_child(self, child: "ResearchQuestion") -> None:
        """添加子RQ"""
        child.parent_id = self.id
        self.children.append(child)

    def get_all_descendants(self) -> List["ResearchQuestion"]:
        """获取所有后代RQ"""
        descendants = []
        for child in self.children:
            descendants.append(child)
            descendants.extend(child.get_all_descendants())
        return descendants

    def get_level_questions(self, level: RQLevel) -> List["ResearchQuestion"]:
        """获取指定层级的所有RQ"""
        if self.level == level:
            return [self]
        results = []
        for child in self.children:
            results.extend(child.get_level_questions(level))
        return results

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "question": self.question,
            "level": self.level.value,
            "description": self.description,
            "parent_id": self.parent_id,
            "children": [c.to_dict() for c in self.children],
            "keywords": self.keywords,
            "status": self.status,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ResearchQuestion":
        """从字典创建"""
        children = [cls.from_dict(c) for c in data.pop("children", [])]
        data["level"] = RQLevel(data["level"])
        obj = cls(**data)
        obj.children = children
        return obj

    def __repr__(self) -> str:
        return f"RQ({self.id}: {self.question})"


@dataclass
class RQTree:
    r"""
    RQ树 - 管理整个研究问题的层级结构

    典型结构：
                    RQ_ROOT
                   /        \
               RQ1(方法)    RQ2(应用)    RQ3(趋势)
              /      \       /      \
          RQ11    RQ12   RQ21    RQ22
         /   \    /   \   /   \    /   \
       RQ111 RQ112 ...
    """

    research_topic: str  # 研究主题
    root_questions: List[ResearchQuestion] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_root_question(self, rq: ResearchQuestion) -> None:
        """添加根级RQ"""
        self.root_questions.append(rq)

    def get_all_questions(self) -> List[ResearchQuestion]:
        """获取所有RQ"""
        all_q = []
        for rq in self.root_questions:
            all_q.append(rq)
            all_q.extend(rq.get_all_descendants())
        return all_q

    def get_question_by_id(self, rq_id: str) -> Optional[ResearchQuestion]:
        """根据ID查找RQ"""
        for rq in self.get_all_questions():
            if rq.id == rq_id:
                return rq
        return None

    def get_questions_by_level(self, level: RQLevel) -> List[ResearchQuestion]:
        """获取指定层级的所有RQ"""
        results = []
        for rq in self.root_questions:
            results.extend(rq.get_level_questions(level))
        return results

    def get_level1_questions(self) -> List[ResearchQuestion]:
        """获取一级RQ（宏观维度）"""
        return self.get_questions_by_level(RQLevel.LEVEL_1)

    def get_level2_questions(self) -> List[ResearchQuestion]:
        """获取二级RQ（中观分析）"""
        return self.get_questions_by_level(RQLevel.LEVEL_2)

    def get_level3_questions(self) -> List[ResearchQuestion]:
        """获取三级RQ（微观深入）"""
        return self.get_questions_by_level(RQLevel.LEVEL_3)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "research_topic": self.research_topic,
            "root_questions": [rq.to_dict() for rq in self.root_questions],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RQTree":
        """从字典创建"""
        root_questions = [ResearchQuestion.from_dict(rq) for rq in data.pop("root_questions", [])]
        return cls(research_topic=data["research_topic"], root_questions=root_questions, metadata=data.get("metadata", {}))

    def print_tree(self) -> str:
        """打印树形结构"""
        lines = [f"Research Topic: {self.research_topic}"]
        for rq in self.root_questions:
            lines.extend(self._print_node(rq, 0))
        return "\n".join(lines)

    def _print_node(self, rq: ResearchQuestion, indent: int) -> List[str]:
        """打印单个节点"""
        prefix = "  " * indent + ("└─ " if indent > 0 else "")
        lines = [f"{prefix}{rq.id}: {rq.question} [{rq.status}]"]
        for child in rq.children:
            lines.extend(self._print_node(child, indent + 1))
        return lines


class RQManager:
    """
    RQ管理器

    职责：
    - 创建和初始化RQ树
    - 管理RQ状态
    - 提供RQ查询接口
    - 生成RQ相关的Prompt
    """

    def __init__(self, workspace_path: str = "./workspace"):
        """
        初始化RQ管理器

        Args:
            workspace_path: 工作区路径
        """
        self.workspace_path = workspace_path
        self.current_tree: Optional[RQTree] = None
        self._templates = self._load_templates()

    def _load_templates(self) -> Dict[str, str]:
        """加载RQ模板"""
        return {
            "level1_methodology": "RQ1: What are the main methodologies used in {topic}?",
            "level1_application": "RQ2: What are the major application domains of {topic}?",
            "level1_trends": "RQ3: What are the research trends and evolution in {topic}?",
            "level2_methods_type": "RQ11: What types of {method_type} methods are used?",
            "level2_methods_comparison": "RQ12: What are the comparative advantages and limitations of different methods?",
            "level2_apps_domains": "RQ21: In which domains is {topic} applied?",
            "level2_apps_challenges": "RQ22: What are the domain-specific challenges?",
            "level3_technical_detail": "RQ111: What are the technical details of {specific_method}?",
            "level3_performance": "RQ112: How does {specific_method} perform in terms of {metric}?",
        }

    async def initialize_from_topic(
        self,
        research_topic: str,
        custom_rqs: Optional[List[Dict[str, Any]]] = None,
    ) -> RQTree:
        """
        根据研究主题初始化RQ树

        Args:
            research_topic: 研究主题
            custom_rqs: 自定义RQ列表（可选）

        Returns:
            RQTree
        """
        if custom_rqs:
            # 使用自定义RQ
            root_questions = [ResearchQuestion.from_dict(rq) for rq in custom_rqs]
        else:
            # 使用默认模板生成
            root_questions = await self._generate_default_rqs(research_topic)

        self.current_tree = RQTree(research_topic=research_topic, root_questions=root_questions)
        await self.save()
        return self.current_tree

    async def _generate_default_rqs(self, research_topic: str) -> List[ResearchQuestion]:
        """
        生成默认的层级RQ

        默认结构：
        RQ1: 方法论维度
          ├── RQ11: 使用了哪些方法类型？
          │   ├── RQ111: 技术细节
          │   └── RQ112: 性能表现
          └── RQ12: 各方法的优缺点对比？
        RQ2: 应用维度
          ├── RQ21: 应用在哪些领域？
          └── RQ22: 各领域面临什么挑战？
        RQ3: 趋势维度
          ├── RQ31: 方法演进趋势
          └── RQ32: 未来研究方向
        """
        rq1 = ResearchQuestion(
            id="RQ1",
            question=f"What are the main methodologies used in {research_topic}?",
            level=RQLevel.LEVEL_1,
            description="Methodology dimension - understanding the technical approaches",
            keywords=["method", "approach", "technique", "algorithm", "framework"],
        )

        rq11 = ResearchQuestion(
            id="RQ11",
            question="What types of methods are used in this domain?",
            level=RQLevel.LEVEL_2,
            description="Categorization of method types",
            keywords=["method type", "category", "class", "family"],
        )

        rq111 = ResearchQuestion(
            id="RQ111",
            question="What are the technical details of specific methods?",
            level=RQLevel.LEVEL_3,
            description="Deep dive into technical implementation",
            keywords=["implementation", "architecture", "technical detail"],
        )

        rq112 = ResearchQuestion(
            id="RQ112",
            question="What are the performance characteristics?",
            level=RQLevel.LEVEL_3,
            description="Performance analysis and metrics",
            keywords=["performance", "accuracy", "efficiency", "scalability"],
        )

        rq12 = ResearchQuestion(
            id="RQ12",
            question="What are the comparative advantages and limitations of different methods?",
            level=RQLevel.LEVEL_2,
            description="Comparative analysis of methods",
            keywords=["advantage", "limitation", "comparison", "trade-off"],
        )

        rq1.add_child(rq11)
        rq1.add_child(rq12)
        rq11.add_child(rq111)
        rq11.add_child(rq112)

        rq2 = ResearchQuestion(
            id="RQ2",
            question=f"What are the major application domains of {research_topic}?",
            level=RQLevel.LEVEL_1,
            description="Application dimension - understanding real-world usage",
            keywords=["application", "domain", "use case", "scenario"],
        )

        rq21 = ResearchQuestion(
            id="RQ21",
            question="In which domains is this research applied?",
            level=RQLevel.LEVEL_2,
            description="Identification of application domains",
            keywords=["domain", "field", "industry", "sector"],
        )

        rq22 = ResearchQuestion(
            id="RQ22",
            question="What are the domain-specific challenges?",
            level=RQLevel.LEVEL_2,
            description="Analysis of challenges in different domains",
            keywords=["challenge", "issue", "problem", "barrier"],
        )

        rq2.add_child(rq21)
        rq2.add_child(rq22)

        rq3 = ResearchQuestion(
            id="RQ3",
            question=f"What are the research trends and evolution in {research_topic}?",
            level=RQLevel.LEVEL_1,
            description="Trends dimension - understanding evolution and future directions",
            keywords=["trend", "evolution", "development", "progress", "future"],
        )

        rq31 = ResearchQuestion(
            id="RQ31",
            question="How have methods evolved over time?",
            level=RQLevel.LEVEL_2,
            description="Temporal evolution analysis",
            keywords=["evolution", "timeline", "progression", "history"],
        )

        rq32 = ResearchQuestion(
            id="RQ32",
            question="What are the emerging research directions?",
            level=RQLevel.LEVEL_2,
            description="Future directions identification",
            keywords=["emerging", "future", "novel", "cutting-edge"],
        )

        rq3.add_child(rq31)
        rq3.add_child(rq32)

        return [rq1, rq2, rq3]

    def get_current_tree(self) -> Optional[RQTree]:
        """获取当前RQ树"""
        return self.current_tree

    def get_question(self, rq_id: str) -> Optional[ResearchQuestion]:
        """根据ID获取RQ"""
        if not self.current_tree:
            return None
        return self.current_tree.get_question_by_id(rq_id)

    def get_level_questions(self, level: RQLevel) -> List[ResearchQuestion]:
        """获取指定层级的所有RQ"""
        if not self.current_tree:
            return []
        return self.current_tree.get_questions_by_level(level)

    def update_question_status(self, rq_id: str, status: str) -> bool:
        """
        更新RQ状态

        Args:
            rq_id: RQ ID
            status: 新状态

        Returns:
            是否成功
        """
        rq = self.get_question(rq_id)
        if rq:
            rq.status = status
            return True
        return False

    def generate_search_queries(self, rq_id: str) -> List[str]:
        """
        根据RQ生成搜索查询

        Args:
            rq_id: RQ ID

        Returns:
            搜索查询列表
        """
        rq = self.get_question(rq_id)
        if not rq:
            return []

        queries = []

        # 基于问题生成查询
        base_query = rq.question.replace("?", "")
        queries.append(base_query)

        # 基于关键词生成查询
        if rq.keywords:
            keyword_query = " ".join(rq.keywords[:5])  # 最多5个关键词
            queries.append(keyword_query)

        # 对于二级和三级RQ，包含父问题的上下文
        if rq.level in [RQLevel.LEVEL_2, RQLevel.LEVEL_3] and rq.parent_id:
            parent_rq = self.get_question(rq.parent_id)
            if parent_rq:
                combined = f"{parent_rq.question} {rq.question}"
                queries.append(combined.replace("?", ""))

        return queries

    def generate_screening_prompt(self, rq_ids: List[str]) -> str:
        """
        根据RQ生成筛选Prompt

        Args:
            rq_ids: 相关的RQ ID列表

        Returns:
            Prompt文本
        """
        rq_descriptions = []
        for rq_id in rq_ids:
            rq = self.get_question(rq_id)
            if rq:
                rq_descriptions.append(f"- {rq.id}: {rq.question}")

        return f"""You are a literature screening expert. Evaluate whether papers are relevant to the following research questions:

{chr(10).join(rq_descriptions)}

For each paper, assess:
1. Topic relevance (1-5): Does the paper directly address any of the RQs?
2. Method relevance (1-5): Is the paper's methodology within scope?
3. Timeliness (1-5): Does the paper reflect recent developments?

Output JSON format:
{{
  "relevant": true/false,
  "confidence": 0.0-1.0,
  "relevance_scores": {{
    "topic": score,
    "method": score,
    "timeliness": score
  }},
  "reasoning": "Brief justification",
  "related_rqs": ["RQ1", "RQ2", ...]
}}"""

    def generate_summary_prompt(self, rq_id: str, cluster_context: str) -> str:
        """
        根据RQ生成摘要Prompt

        Args:
            rq_id: 目标RQ ID
            cluster_context: 聚类上下文信息

        Returns:
            Prompt文本
        """
        rq = self.get_question(rq_id)
        if not rq:
            return ""

        level_prompts = {
            RQLevel.LEVEL_1: """Generate a comprehensive summary addressing the research question: {rq}

Cluster context: {context}

Please structure your response as:
1. Overview of the domain
2. Main categories/themes
3. Cross-cutting trends
4. Research gaps and opportunities""",
            RQLevel.LEVEL_2: """Generate a detailed analysis addressing: {rq}

Cluster context: {context}

Please structure your response as:
1. Specific items in this category
2. Comparative analysis (table format)
3. Technical details
4. Advantages and limitations""",
            RQLevel.LEVEL_3: """Generate an in-depth analysis addressing: {rq}

Cluster context: {context}

Please structure your response as:
1. Technical implementation details
2. Performance characteristics with data
3. Specific advantages and limitations
4. Applicable scenarios""",
        }

        template = level_prompts.get(rq.level, level_prompts[RQLevel.LEVEL_2])
        return template.format(rq=rq.question, context=cluster_context)

    async def save(self) -> None:
        """保存RQ树"""
        if self.current_tree:
            import aiofiles
            from pathlib import Path

            path = Path(self.workspace_path) / "rq_tree.json"
            path.parent.mkdir(parents=True, exist_ok=True)

            async with aiofiles.open(path, "w", encoding="utf-8") as f:
                await f.write(json.dumps(self.current_tree.to_dict(), ensure_ascii=False, indent=2))

    async def load(self) -> Optional[RQTree]:
        """加载RQ树"""
        import aiofiles
        from pathlib import Path

        path = Path(self.workspace_path) / "rq_tree.json"
        if not path.exists():
            return None

        async with aiofiles.open(path, encoding="utf-8") as f:
            data = json.loads(await f.read())
            self.current_tree = RQTree.from_dict(data)
            return self.current_tree

    def export_for_report(self) -> Dict[str, Any]:
        """导出用于报告的RQ信息"""
        if not self.current_tree:
            return {}

        return {
            "research_topic": self.current_tree.research_topic,
            "structure": self.current_tree.print_tree(),
            "level1": [rq.to_dict() for rq in self.current_tree.get_level1_questions()],
            "level2": [rq.to_dict() for rq in self.current_tree.get_level2_questions()],
            "level3": [rq.to_dict() for rq in self.current_tree.get_level3_questions()],
        }
