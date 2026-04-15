"""
Coordinator Agent - 协调者Agent
负责全局任务调度、状态管理、质量门控
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
import logging

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from .agent import BaseAgent, AgentConfig, AgentResult
from .workspace import SharedWorkspace
from .rq_manager import RQManager, RQLevel, ResearchQuestion


class Stage(Enum):
    """系统处理阶段"""
    INITIALIZATION = "initialization"
    SEARCH = "search"
    SCREEN = "screen"
    CLUSTER = "cluster"
    SUMMARY = "summary"
    FINALIZATION = "finalization"
    COMPLETED = "completed"


class QualityGate(Enum):
    """质量门控节点"""
    POST_SEARCH = "post_search"  # 搜索后确认文献池
    POST_SCREEN = "post_screen"  # 筛选后确认相关文献
    POST_CLUSTER = "post_cluster"  # 聚类后确认主题结构
    POST_SUMMARY = "post_summary"  # 摘要后确认综述质量


@dataclass
class TaskState:
    """任务状态"""
    current_stage: Stage = Stage.INITIALIZATION
    completed_stages: List[Stage] = field(default_factory=list)
    pending_stages: List[Stage] = field(default_factory=lambda: [
        Stage.SEARCH, Stage.SCREEN, Stage.CLUSTER, Stage.SUMMARY
    ])
    metrics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    checkpoints: List[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage.value,
            "completed_stages": [s.value for s in self.completed_stages],
            "pending_stages": [s.value for s in self.pending_stages],
            "metrics": self.metrics,
            "errors": self.errors,
            "checkpoints": self.checkpoints,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }


@dataclass
class CoordinatorDecision:
    """协调者决策"""
    action: str  # continue, human_review, rollback, complete, skip
    target_agent: Optional[str] = None
    task_description: Optional[str] = None
    reason: str = ""
    confidence: float = 1.0


class Coordinator(BaseAgent):
    """
    协调者Agent

    职责：
    - 任务调度与状态管理
    - 层级RQ管理
    - 质量门控（人工审核节点）
    - 工作流编排
    - 错误处理与恢复
    """

    def __init__(
        self,
        workspace: SharedWorkspace,
        rq_manager: RQManager,
        llm_client: Optional[ChatOpenAI] = None,
        config: Optional[AgentConfig] = None,
    ):
        """
        初始化协调者

        Args:
            workspace: 共享工作区
            rq_manager: RQ管理器
            llm_client: LLM客户端
            config: Agent配置
        """
        config = config or AgentConfig(
            name="Coordinator",
            description="Coordinates all agents in the literature review system",
            model="glm-4-plus",
            temperature=0.3,
        )
        super().__init__(config, workspace, llm_client)

        self.rq_manager = rq_manager
        self.task_state = TaskState()
        self.agents: Dict[str, BaseAgent] = {}
        self.quality_gates: Dict[QualityGate, bool] = {
            gate: False for gate in QualityGate
        }
        self.human_review_callbacks: Dict[QualityGate, Callable] = {}

    def register_agent(self, agent: BaseAgent) -> None:
        """注册专业Agent"""
        self.agents[agent.name] = agent
        self.log_progress(f"Registered agent: {agent.name}")

    def register_human_review_callback(
        self, gate: QualityGate, callback: Callable
    ) -> None:
        """注册人工审核回调"""
        self.human_review_callbacks[gate] = callback

    def validate_input(self, **kwargs) -> bool:
        """验证输入参数"""
        required = ["research_topic"]
        return all(k in kwargs for k in required)

    async def execute(self, **kwargs) -> AgentResult:
        """
        执行协调任务 - 主工作流

        Args:
            **kwargs: 执行参数，至少包含 research_topic

        Returns:
            AgentResult
        """
        research_topic = kwargs.get("research_topic")
        auto_mode = kwargs.get("auto_mode", False)  # 自动模式，跳过人工审核

        try:
            # 初始化RQ树
            self.log_progress(f"Initializing RQ tree for: {research_topic}")
            await self.rq_manager.initialize_from_topic(research_topic)
            await self._advance_stage(Stage.INITIALIZATION)

            # 创建初始检查点
            checkpoint_name = f"init_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            checkpoint_path = await self.workspace.create_checkpoint(checkpoint_name)
            self.task_state.checkpoints.append(checkpoint_path)
            self.log_progress(f"Created initial checkpoint: {checkpoint_path}")

            # 执行主工作流
            while not self._is_workflow_complete():
                decision = await self._make_decision()

                if decision.action == "continue":
                    await self._execute_next_stage(decision)
                elif decision.action == "human_review":
                    if auto_mode:
                        # 自动模式：跳过人工审核，记录并继续
                        self.log_progress(f"Auto mode: Skipping human review at {self.task_state.current_stage}")
                        await self._approve_quality_gate()
                    else:
                        # 等待人工审核
                        await self._request_human_review(decision)
                elif decision.action == "rollback":
                    await self._execute_rollback(decision)
                elif decision.action == "skip":
                    await self._skip_current_stage()
                elif decision.action == "complete":
                    break

            # 完成工作流
            return await self._finalize_workflow()

        except KeyboardInterrupt:
            # 重新抛出 KeyboardInterrupt，让上层处理
            raise
        except Exception as e:
            error_msg = f"Workflow execution failed: {str(e)}"
            self.log_progress(error_msg, "error")
            self.task_state.errors.append(error_msg)
            return self._create_result(success=False, errors=[error_msg])

    async def _make_decision(self) -> CoordinatorDecision:
        """
        做出下一步决策

        Returns:
            CoordinatorDecision
        """
        current_stage = self.task_state.current_stage

        # 检查是否需要人工审核
        quality_gate = self._get_quality_gate_for_stage(current_stage)
        if quality_gate and not self.quality_gates[quality_gate]:
            return CoordinatorDecision(
                action="human_review",
                reason=f"Quality gate {quality_gate.value} requires human review",
            )

        # 检查是否有待处理的阶段
        if self.task_state.pending_stages:
            next_stage = self.task_state.pending_stages[0]
            agent_name = self._get_agent_for_stage(next_stage)

            return CoordinatorDecision(
                action="continue",
                target_agent=agent_name,
                task_description=f"Execute {next_stage.value} stage",
                reason=f"Proceeding to {next_stage.value} stage",
            )

        # 工作流完成
        return CoordinatorDecision(
            action="complete",
            reason="All stages completed",
        )

    async def _execute_next_stage(self, decision: CoordinatorDecision) -> None:
        """执行下一个阶段"""
        stage = self.task_state.pending_stages.pop(0)
        agent = self.agents.get(decision.target_agent)

        if not agent:
            raise ValueError(f"Agent not found: {decision.target_agent}")

        self.log_progress(f"Executing {stage.value} stage with {agent.name}")

        # 准备执行参数
        exec_params = await self._prepare_stage_params(stage)

        # 执行Agent
        result = await agent.run(**exec_params)

        # 处理结果
        if result.success:
            self.task_state.completed_stages.append(stage)
            self.task_state.metrics[f"{stage.value}_metrics"] = result.metrics
            await self._advance_stage(stage)
            self.log_progress(f"Completed {stage.value} stage: {result.metrics}")
        else:
            self.task_state.errors.extend(result.errors)
            raise RuntimeError(f"Stage {stage.value} failed: {result.errors}")

    async def _prepare_stage_params(self, stage: Stage) -> Dict[str, Any]:
        """为阶段准备执行参数"""
        params = {}

        if stage == Stage.SEARCH:
            params["research_topic"] = self.rq_manager.current_tree.research_topic
            params["year_range"] = (2018, 2025)  # 默认值
            params["max_results"] = 500

        elif stage == Stage.SCREEN:
            params["rq_ids"] = [rq.id for rq in self.rq_manager.get_level_questions(RQLevel.LEVEL_1)]
            params["threshold"] = 0.7  # NF阈值

        elif stage == Stage.CLUSTER:
            params["method"] = "HDBSCAN"
            params["min_cluster_size"] = 5

        elif stage == Stage.SUMMARY:
            params["rq_tree"] = self.rq_manager.current_tree

        return params

    async def _request_human_review(self, decision: CoordinatorDecision) -> None:
        """请求人工审核"""
        gate = self._get_quality_gate_for_stage(self.task_state.current_stage)

        self.log_progress(f"Requesting human review at {gate.value}")

        # 调用注册的回调函数
        callback = self.human_review_callbacks.get(gate)
        if callback:
            # 准备审核数据
            review_data = await self._prepare_review_data(gate)
            approved = await callback(gate, review_data)

            if approved:
                await self._approve_quality_gate()
            else:
                # 人工拒绝，可能需要回滚
                raise RuntimeError(f"Human review rejected at {gate.value}")
        else:
            # 没有回调，自动批准（用于测试）
            self.log_progress("No human review callback, auto-approving", "warning")
            await self._approve_quality_gate()

    async def _prepare_review_data(self, gate: QualityGate) -> Dict[str, Any]:
        """准备人工审核数据"""
        if gate == QualityGate.POST_SEARCH:
            return {
                "stage": "search",
                "literature_count": self.workspace.get_literature_count(),
                "sample_papers": (await self.workspace.get_literature())[:10],
            }
        elif gate == QualityGate.POST_SCREEN:
            relevant = await self.workspace.get_literature(
                filters={"relevance_score": lambda x: x is not None}
            )
            return {
                "stage": "screen",
                "relevant_count": len(relevant),
                "sample_relevant": relevant[:10],
            }
        elif gate == QualityGate.POST_CLUSTER:
            clusters = await self.workspace.get_clusters()
            return {
                "stage": "cluster",
                "cluster_count": len(clusters),
                "clusters": clusters,
            }
        elif gate == QualityGate.POST_SUMMARY:
            summaries = await self.workspace.get_all_summaries()
            return {
                "stage": "summary",
                "summaries": summaries,
            }
        return {}

    async def _approve_quality_gate(self) -> None:
        """批准质量门控"""
        gate = self._get_quality_gate_for_stage(self.task_state.current_stage)
        if gate:
            self.quality_gates[gate] = True
            self.log_progress(f"Approved quality gate: {gate.value}")

    async def _execute_rollback(self, decision: CoordinatorDecision) -> None:
        """执行回滚"""
        self.log_progress(f"Rolling back: {decision.reason}")

        # 确定回滚目标
        if self.task_state.completed_stages:
            # 回到上一个阶段
            last_stage = self.task_state.completed_stages.pop()
            self.task_state.pending_stages.insert(0, last_stage)
            self.task_state.pending_stages.insert(0, self.task_state.current_stage)

            # 恢复检查点
            if self.task_state.checkpoints:
                last_checkpoint = self.task_state.checkpoints[-1]
                await self.workspace.restore_checkpoint(last_checkpoint)

            await self._advance_stage(last_stage)

    async def _skip_current_stage(self) -> None:
        """跳过当前阶段"""
        self.log_progress(f"Skipping stage: {self.task_state.current_stage}")
        if self.task_state.pending_stages:
            self.task_state.pending_stages.pop(0)
        if self.task_state.pending_stages:
            await self._advance_stage(self.task_state.pending_stages[0])

    async def _advance_stage(self, stage: Stage) -> None:
        """前进到指定阶段"""
        self.task_state.current_stage = stage
        self.log_progress(f"Advanced to stage: {stage.value}")

    def _get_quality_gate_for_stage(self, stage: Stage) -> Optional[QualityGate]:
        """获取阶段对应的质量门控"""
        mapping = {
            Stage.SEARCH: QualityGate.POST_SEARCH,
            Stage.SCREEN: QualityGate.POST_SCREEN,
            Stage.CLUSTER: QualityGate.POST_CLUSTER,
            Stage.SUMMARY: QualityGate.POST_SUMMARY,
        }
        return mapping.get(stage)

    def _get_agent_for_stage(self, stage: Stage) -> str:
        """获取处理阶段的Agent"""
        mapping = {
            Stage.SEARCH: "SearchAgent",
            Stage.SCREEN: "ScreenAgent",
            Stage.CLUSTER: "ClusterAgent",
            Stage.SUMMARY: "SummaryAgent",
        }
        return mapping.get(stage, "")

    def _is_workflow_complete(self) -> bool:
        """检查工作流是否完成"""
        return (
            len(self.task_state.pending_stages) == 0
            and self.task_state.current_stage != Stage.INITIALIZATION
        )

    async def _finalize_workflow(self) -> AgentResult:
        """完成工作流"""
        self.task_state.current_stage = Stage.COMPLETED
        self.task_state.completed_at = datetime.now().isoformat()

        # 生成最终报告
        report = await self._generate_final_report()
        await self.workspace.save("final_report", report, agent=self.name, stage="finalization")

        self.log_progress("Workflow completed successfully")

        return self._create_result(
            success=True,
            data={"report": report, "task_state": self.task_state.to_dict()},
            metrics=self.task_state.metrics,
        )

    async def _generate_final_report(self) -> Dict[str, Any]:
        """生成最终报告"""
        return {
            "research_topic": self.rq_manager.current_tree.research_topic,
            "workflow_summary": {
                "started_at": self.task_state.started_at,
                "completed_at": self.task_state.completed_at,
                "stages_completed": [s.value for s in self.task_state.completed_stages],
                "total_errors": len(self.task_state.errors),
            },
            "literature_summary": {
                "total_papers": self.workspace.get_literature_count(),
                "clusters": len(await self.workspace.get_clusters()),
                "summaries": len(await self.workspace.get_all_summaries()),
            },
            "rq_structure": self.rq_manager.export_for_report(),
            "quality_gates": {gate.value: passed for gate, passed in self.quality_gates.items()},
        }

    def get_progress_summary(self) -> Dict[str, Any]:
        """获取进度摘要"""
        return {
            "current_stage": self.task_state.current_stage.value,
            "progress_percentage": len(self.task_state.completed_stages) / max(
                len(self.task_state.completed_stages) + len(self.task_state.pending_stages), 1
            ) * 100,
            "completed_stages": [s.value for s in self.task_state.completed_stages],
            "pending_stages": [s.value for s in self.task_state.pending_stages],
            "errors": self.task_state.errors,
            "metrics": self.task_state.metrics,
        }

    async def resume_from_checkpoint(self, checkpoint_name: str) -> bool:
        """从检查点恢复"""
        self.log_progress(f"Resuming from checkpoint: {checkpoint_name}")
        success = await self.workspace.restore_checkpoint(checkpoint_name)
        if success:
            await self.rq_manager.load()
            self.log_progress("Successfully resumed from checkpoint")
        return success

    def __repr__(self) -> str:
        return f"Coordinator(stage={self.task_state.current_stage.value}, agents={len(self.agents)})"
