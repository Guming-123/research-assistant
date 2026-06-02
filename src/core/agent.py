"""
Base Agent class for all specialized agents in the system.
所有专业Agent的基类
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio
import json
import logging
import yaml
from pathlib import Path

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from .workspace import SharedWorkspace
from ..utils.llm import get_llm_client
from ..utils.exceptions import LLMError, ValidationError

# 全局 LLM 并发信号量（控制同时向 GLM API 发送的请求数）
_LLM_SEMAPHORE = asyncio.Semaphore(15)
logger = logging.getLogger(__name__)

_CONFIG_CACHE: Optional[Dict[str, Any]] = None


def get_config() -> Dict[str, Any]:
    """读取并缓存 config.yaml"""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        if cfg_path.exists():
            _CONFIG_CACHE = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        else:
            _CONFIG_CACHE = {}
    return _CONFIG_CACHE


def get_agent_model(agent_name: str, default: str = "glm-4-flash") -> str:
    """从 config.yaml 读取指定 agent 的模型名"""
    cfg = get_config()
    return cfg.get("agents", {}).get(agent_name, {}).get("model", default)


@dataclass
class AgentConfig:
    """Agent配置类"""

    name: str
    description: str
    model: str = "gpt-4o"
    temperature: float = 0.7
    max_tokens: int = 8000
    enable_streaming: bool = False
    retry_attempts: int = 3
    timeout: int = 120


@dataclass
class AgentResult:
    """Agent执行结果"""

    agent_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "agent_name": self.agent_name,
            "success": self.success,
            "data": self.data,
            "errors": self.errors,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
        }


class BaseAgent(ABC):
    """
    所有专业Agent的抽象基类

    定义了Agent的基本接口和通用功能：
    - 生命周期管理（初始化、执行、清理）
    - LLM调用
    - 工作区访问
    - 结果记录
    """

    def __init__(
        self,
        config: AgentConfig,
        workspace: SharedWorkspace,
        llm_client: Optional[ChatOpenAI] = None,
    ):
        """
        初始化Agent

        Args:
            config: Agent配置
            workspace: 共享工作区
            llm_client: LLM客户端（可选，默认使用全局配置）
        """
        self.config = config
        self.workspace = workspace
        self.llm = llm_client or get_llm_client(
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        self.logger = logging.getLogger(f"Agent.{config.name}")
        self._state: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        """Agent名称"""
        return self.config.name

    @property
    def state(self) -> Dict[str, Any]:
        """获取Agent内部状态"""
        return self._state.copy()

    def set_state(self, key: str, value: Any) -> None:
        """设置Agent内部状态"""
        self._state[key] = value
        self.logger.debug(f"State updated: {key} = {value}")

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """
        执行Agent的主要任务（抽象方法，子类必须实现）

        Args:
            **kwargs: 执行参数

        Returns:
            AgentResult: 执行结果
        """
        pass

    @abstractmethod
    def validate_input(self, **kwargs) -> bool:
        """
        验证输入参数（抽象方法，子类必须实现）

        Args:
            **kwargs: 待验证的参数

        Returns:
            bool: 验证是否通过
        """
        pass

    async def _call_llm(
        self,
        messages: List[BaseMessage],
        response_format: Optional[Dict[str, str]] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        调用LLM的通用方法（带并发信号量）

        Args:
            messages: 消息列表
            response_format: 响应格式（用于结构化输出）
            max_tokens: 覆盖本次调用的最大输出 token 数（可选）
            **kwargs: 额外参数

        Returns:
            str: LLM响应
        """
        async with _LLM_SEMAPHORE:
            try:
                if response_format:
                    kwargs["response_format"] = {"type": "json_object"}

                # 按需覆盖 max_tokens
                llm = self.llm.bind(max_tokens=max_tokens) if max_tokens else self.llm

                response = await llm.ainvoke(messages, **kwargs)
                return response.content

            except (ConnectionError, TimeoutError) as e:
                self.log_progress(f"LLM连接失败: {e}", "error")
                raise LLMError(f"LLM connection error: {e}") from e
            except json.JSONDecodeError as e:
                self.log_progress(f"LLM返回的JSON格式错误: {e}", "error")
                raise LLMError(f"LLM returned invalid JSON: {e}") from e
            except Exception as e:
                self.log_progress(f"LLM调用失败: {e}", "error")
                raise LLMError(f"LLM invocation failed: {e}") from e

    def _create_system_prompt(self, prompt_template: str, **kwargs) -> SystemMessage:
        """创建系统提示消息"""
        return SystemMessage(content=prompt_template.format(**kwargs))

    def _create_user_prompt(self, prompt_template: str, **kwargs) -> HumanMessage:
        """创建用户提示消息"""
        return HumanMessage(content=prompt_template.format(**kwargs))

    def log_progress(self, message: str, level: str = "info") -> None:
        """记录进度日志"""
        log_func = getattr(self.logger, level, self.logger.info)
        log_func(f"[{self.name}] {message}")

    async def _save_to_workspace(
        self, key: str, data: Any, stage: str = "intermediate"
    ) -> None:
        """
        保存数据到工作区

        Args:
            key: 数据键
            data: 数据内容
            stage: 阶段标识
        """
        await self.workspace.save(key, data, agent=self.name, stage=stage)
        self.log_progress(f"Saved data to workspace: {key}")

    async def _load_from_workspace(self, key: str) -> Optional[Any]:
        """
        从工作区加载数据

        Args:
            key: 数据键

        Returns:
            数据内容，不存在则返回None
        """
        data = await self.workspace.load(key)
        if data is not None:
            self.log_progress(f"Loaded data from workspace: {key}")
        return data

    def _create_result(
        self,
        success: bool,
        data: Dict[str, Any] = None,
        errors: List[str] = None,
        metrics: Dict[str, Any] = None,
    ) -> AgentResult:
        """
        创建执行结果

        Args:
            success: 是否成功
            data: 结果数据
            errors: 错误列表
            metrics: 指标数据

        Returns:
            AgentResult
        """
        return AgentResult(
            agent_name=self.name,
            success=success,
            data=data or {},
            errors=errors or [],
            metrics=metrics or {},
        )

    async def run(self, **kwargs) -> AgentResult:
        """
        运行Agent（带验证和错误处理）

        Args:
            **kwargs: 执行参数

        Returns:
            AgentResult: 执行结果
        """
        self.log_progress(f"Starting execution with params: {list(kwargs.keys())}")

        # 验证输入
        if not self.validate_input(**kwargs):
            error_msg = "Input validation failed"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

        try:
            # 执行主要任务
            result = await self.execute(**kwargs)

            # 记录结果
            if result.success:
                self.log_progress("Execution completed successfully")
            else:
                self.log_progress(f"Execution completed with errors: {result.errors}", "warning")

            return result

        except KeyboardInterrupt:
            # 重新抛出 KeyboardInterrupt，让上层处理
            raise
        except (LLMError, ValidationError) as e:
            # 已知的业务异常，直接记录
            error_msg = f"{type(e).__name__}: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])
        except (asyncio.TimeoutError, TimeoutError) as e:
            error_msg = f"Execution timeout: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])
        except Exception as e:
            # 未预期的异常，记录详细信息便于调试
            error_msg = f"Unexpected error during execution: {type(e).__name__}: {str(e)}"
            self.log_progress(error_msg, "error")
            return self._create_result(success=False, errors=[error_msg])

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, model={self.config.model})"
