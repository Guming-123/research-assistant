"""Core components for the Multi-Agent Literature Review System"""

from .agent import BaseAgent, AgentConfig
from .coordinator import Coordinator
from .workspace import SharedWorkspace
from .rq_manager import RQManager, ResearchQuestion

__all__ = [
    "BaseAgent",
    "AgentConfig",
    "Coordinator",
    "SharedWorkspace",
    "RQManager",
    "ResearchQuestion",
]
