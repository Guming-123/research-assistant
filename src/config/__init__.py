"""
Configuration management for Multi-Agent Literature Review System
配置管理模块
"""

import os
import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """Agent配置类"""
    name: str
    description: str = ""
    model: str = "glm-5"
    temperature: float = 0.7
    max_tokens: int = 8000
    enable_streaming: bool = False
    timeout: int = 120
    retry_attempts: int = 3


@dataclass
class SearchConfig(AgentConfig):
    """Search Agent配置"""
    default_databases: list = field(default_factory=lambda: ["semantic_scholar", "arxiv"])
    enable_pdf_download: bool = False
    max_concurrent_requests: int = 5
    default_year_start: int = 2018
    default_year_end: int = 2025
    default_max_results: int = 500
    rate_limit_per_second: int = 10


@dataclass
class ScreenConfig(AgentConfig):
    """Screen Agent配置"""
    chunk_size: int = 512
    chunk_overlap: int = 50
    default_nf_threshold: float = 0.3
    enable_llm_screening: bool = True
    similarity_threshold: float = 0.5
    top_k_chunks: int = 50
    llm_threshold_min: float = 0.2
    llm_threshold_max: float = 0.8
    # 两阶段筛选模型配置
    screening_model: str = "glm-5"  # 初筛模型（快速、低成本）
    refinement_model: Optional[str] = "glm-5"  # 精确判定模型（可选）


@dataclass
class ClusterConfig(AgentConfig):
    """Cluster Agent配置"""
    method: str = "hdbscan"
    min_cluster_size: int = 5
    min_samples: int = 3
    dimensionality_reduction: str = "tsne"
    n_components: int = 2
    perplexity: int = 30
    max_clusters: int = 20


@dataclass
class SummaryConfig(AgentConfig):
    """Summary Agent配置"""
    max_papers_per_cluster: int = 20
    include_methodology: bool = True
    include_applications: bool = True


@dataclass
class SystemConfig:
    """系统配置类"""
    name: str = "Multi-Agent Literature Review System"
    version: str = "1.0.0"

    # 工作区配置
    workspace_path: str = "./workspace"
    auto_save: bool = True
    checkpoint_interval: int = 300
    max_checkpoints: int = 10

    # LLM配置
    llm_model: str = "glm-5"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4000
    llm_timeout: int = 120
    llm_retry_attempts: int = 3

    # Embedding配置
    embedding_model: str = "local-zh"  # 默认使用本地中文模型
    embedding_dimensions: int = 768  # 本地模型通常是768维
    embedding_batch_size: int = 32  # 本地模型批次大小

    # API配置
    s2_base_url: str = "https://api.semanticscholar.org/graph/v1"
    s2_rate_limit: int = 100
    arxiv_base_url: str = "http://export.arxiv.org/api/query"
    arxiv_rate_limit: int = 10

    # 日志配置
    log_level: str = "INFO"
    log_file: str = "./workspace/logs/system.log"

    # 报告配置
    report_format: str = "markdown"
    report_language: str = "zh-CN"
    report_output_directory: str = "./workspace/reports"


class ConfigLoader:
    """
    配置加载器

    负责从YAML文件和环境变量加载配置
    """

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置加载器

        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path or os.getenv("CONFIG_PATH", "config.yaml")
        self._config: Dict[str, Any] = {}
        self._system_config: Optional[SystemConfig] = None

    def load(self) -> SystemConfig:
        """
        加载配置

        Returns:
            SystemConfig
        """
        # 加载YAML配置文件
        self._load_yaml_config()

        # 加载环境变量（覆盖YAML配置）
        self._load_env_config()

        # 创建SystemConfig对象
        self._system_config = self._create_system_config()

        logger.info(f"Configuration loaded from {self.config_path}")
        return self._system_config

    def _load_yaml_config(self) -> None:
        """加载YAML配置文件"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {self.config_path}, using defaults")
            self._config = {}
            return

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f) or {}
            logger.debug(f"Loaded YAML config from {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to load YAML config: {e}, using defaults")
            self._config = {}

    def _load_env_config(self) -> None:
        """加载环境变量配置"""
        env_mappings = {
            "OPENAI_API_KEY": "llm.api_key",
            "OPENAI_BASE_URL": "llm.base_url",
            "OPENAI_MODEL": "llm.model",
            "EMBEDDING_MODEL": "embedding.model",
            "WORKSPACE_PATH": "workspace.path",
            "LOG_LEVEL": "logging.level",
            "LOG_FILE": "logging.file",
        }

        for env_key, config_path in env_mappings.items():
            value = os.getenv(env_key)
            if value:
                self._set_nested_value(config_path, value)
                logger.debug(f"Loaded env var: {env_key}")

    def _set_nested_value(self, path: str, value: Any) -> None:
        """
        设置嵌套配置值

        Args:
            path: 配置路径，如 "llm.model"
            value: 值
        """
        keys = path.split('.')
        current = self._config

        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]

        current[keys[-1]] = value

    def _create_system_config(self) -> SystemConfig:
        """从配置字典创建SystemConfig对象"""
        # 获取各部分配置
        system = self._config.get("system", {})
        workspace = self._config.get("workspace", {})
        llm = self._config.get("llm", {})
        embedding = self._config.get("embedding", {})
        api = self._config.get("api", {})
        logging_config = self._config.get("logging", {})
        report = self._config.get("report", {})

        # 创建配置对象
        return SystemConfig(
            name=system.get("name", "Multi-Agent Literature Review System"),
            version=system.get("version", "1.0.0"),

            # 工作区
            workspace_path=workspace.get("path", "./workspace"),
            auto_save=workspace.get("auto_save", True),
            checkpoint_interval=workspace.get("checkpoint_interval", 300),
            max_checkpoints=workspace.get("max_checkpoints", 10),

            # LLM
            llm_model=llm.get("default_model", "glm-5"),
            llm_temperature=llm.get("default_temperature", 0.7),
            llm_max_tokens=llm.get("default_max_tokens", 8000),
            llm_timeout=llm.get("timeout", 120),
            llm_retry_attempts=llm.get("retry_attempts", 3),

            # Embedding
            embedding_model=embedding.get("model", "local-zh"),
            embedding_dimensions=embedding.get("dimensions", 768),
            embedding_batch_size=embedding.get("batch_size", 32),

            # API
            s2_base_url=api.get("semantic_scholar", {}).get("base_url", "https://api.semanticscholar.org/graph/v1"),
            s2_rate_limit=api.get("semantic_scholar", {}).get("rate_limit", 100),
            arxiv_base_url=api.get("arxiv", {}).get("base_url", "http://export.arxiv.org/api/query"),
            arxiv_rate_limit=api.get("arxiv", {}).get("rate_limit", 10),

            # 日志
            log_level=logging_config.get("level", "INFO"),
            log_file=logging_config.get("file", "./workspace/logs/system.log"),

            # 报告
            report_format=report.get("format", "markdown"),
            report_language=report.get("language", "zh-CN"),
            report_output_directory=report.get("output_directory", "./workspace/reports"),
        )

    def get_agent_config(self, agent_name: str) -> AgentConfig:
        """
        获取特定Agent的配置

        Args:
            agent_name: Agent名称 (search, screen, cluster, summary)

        Returns:
            对应的Agent配置对象
        """
        agents_config = self._config.get("agents", {})
        agent_config = agents_config.get(agent_name, {})

        base_config = {
            "model": agent_config.get("model", self._system_config.llm_model),
            "temperature": agent_config.get("temperature", self._system_config.llm_temperature),
            "max_tokens": agent_config.get("max_tokens", self._system_config.llm_max_tokens),
            "timeout": self._system_config.llm_timeout,
            "retry_attempts": self._system_config.llm_retry_attempts,
        }

        if agent_name == "search":
            return SearchConfig(
                name="SearchAgent",
                **base_config,
                default_databases=agent_config.get("default_databases", ["semantic_scholar", "arxiv"]),
                enable_pdf_download=agent_config.get("enable_pdf_download", False),
                default_year_start=self._config.get("search", {}).get("default_year_start", 2018),
                default_year_end=self._config.get("search", {}).get("default_year_end", 2025),
                default_max_results=self._config.get("search", {}).get("default_max_results", 500),
                rate_limit_per_second=self._config.get("search", {}).get("rate_limit_per_second", 10),
            )
        elif agent_name == "screen":
            # 获取筛选配置
            screening_config = self._config.get("screening", {})
            return ScreenConfig(
                name="ScreenAgent",
                **base_config,
                chunk_size=agent_config.get("chunk_size", 512),
                chunk_overlap=agent_config.get("chunk_overlap", 50),
                default_nf_threshold=screening_config.get("nf_threshold", 0.7),
                enable_llm_screening=agent_config.get("enable_llm_screening", True),
                similarity_threshold=screening_config.get("similarity_threshold", 0.5),
                top_k_chunks=screening_config.get("top_k_chunks", 10),
                llm_threshold_min=screening_config.get("llm_threshold_min", 0.5),
                llm_threshold_max=screening_config.get("llm_threshold_max", 0.8),
                # 模型分层配置
                screening_model=screening_config.get("screening_model", "glm-5"),
                refinement_model=screening_config.get("refinement_model", "glm-5"),
            )
        elif agent_name == "cluster":
            return ClusterConfig(
                name="ClusterAgent",
                **base_config,
                method=agent_config.get("method", "hdbscan"),
                min_cluster_size=agent_config.get("min_cluster_size", 5),
                min_samples=agent_config.get("min_samples", 3),
                dimensionality_reduction=agent_config.get("dimensionality_reduction", "tsne"),
                n_components=agent_config.get("n_components", 2),
                perplexity=agent_config.get("perplexity", 30),
            )
        elif agent_name == "summary":
            return SummaryConfig(
                name="SummaryAgent",
                **base_config,
                max_papers_per_cluster=agent_config.get("max_papers_per_cluster", 20),
                include_methodology=agent_config.get("include_methodology", True),
                include_applications=agent_config.get("include_applications", True),
            )
        else:
            return AgentConfig(name=agent_name, **base_config)

    @property
    def config(self) -> SystemConfig:
        """获取系统配置"""
        if self._system_config is None:
            self.load()
        return self._system_config


# 全局配置加载器实例
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_path: Optional[str] = None) -> ConfigLoader:
    """
    获取全局配置加载器实例

    Args:
        config_path: 配置文件路径

    Returns:
        ConfigLoader实例
    """
    global _config_loader
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    return _config_loader


def load_config(config_path: Optional[str] = None) -> SystemConfig:
    """
    加载系统配置（便捷函数）

    Args:
        config_path: 配置文件路径

    Returns:
        SystemConfig
    """
    return get_config_loader(config_path).load()


def get_agent_config(agent_name: str) -> AgentConfig:
    """
    获取Agent配置（便捷函数）

    Args:
        agent_name: Agent名称

    Returns:
        Agent配置
    """
    return get_config_loader().get_agent_config(agent_name)