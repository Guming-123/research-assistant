"""
Command Line Interface for Multi-Agent Literature Review System
命令行界面
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
import argparse
import logging

from .core import Coordinator, SharedWorkspace, RQManager
from .agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent
from .utils.llm import get_llm_client

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LiteratureReviewCLI:
    """文献综述系统CLI"""

    def __init__(
        self,
        workspace_path: str = "./workspace",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """
        初始化CLI

        Args:
            workspace_path: 工作区路径
            api_key: API密钥
            base_url: API基础URL
        """
        self.workspace_path = workspace_path
        self.workspace = SharedWorkspace(workspace_path)
        self.rq_manager = RQManager(workspace_path, workspace=self.workspace)
        self._initialized = False
        self._init_lock = asyncio.Lock()  # 并发锁保护

    async def _ensure_initialized(self) -> None:
        """
        延迟初始化（解决async问题）

        添加双重检查锁模式，防止并发重复初始化
        """
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            try:
                await self.workspace.load_all()
                await self.rq_manager.load()
                # 如果没有设置 topic，从 RQ 树恢复最近使用的 topic
                if not self.workspace.topic and self.rq_manager.current_tree:
                    self.workspace.set_topic(self.rq_manager.current_tree.research_topic)
                logger.info("Loaded existing workspace data")
            except Exception as e:
                logger.debug(f"No existing workspace data: {e}")
            self._initialized = True

    async def run_review(
        self,
        research_topic: str,
        year_range: tuple = (2018, 2025),
        max_results: int = 500,
        auto_mode: bool = True,
    ) -> dict:
        """
        运行完整的文献综述流程

        Args:
            research_topic: 研究主题
            year_range: 年份范围
            max_results: 最大结果数
            auto_mode: 自动模式（跳过人工审核）

        Returns:
            执行结果
        """
        await self._ensure_initialized()
        logger.info(f"Starting literature review for: {research_topic}")

        # 创建Coordinator
        coordinator = Coordinator(
            workspace=self.workspace,
            rq_manager=self.rq_manager,
        )

        # 注册各专业Agent
        coordinator.register_agent(SearchAgent(self.workspace))
        coordinator.register_agent(ScreenAgent(self.workspace, self.rq_manager))
        coordinator.register_agent(ClusterAgent(self.workspace))
        coordinator.register_agent(SummaryAgent(self.workspace))

        # 注册人工审核回调（即使在自动模式下）
        async def human_review_callback(gate, review_data):
            logger.info(f"Human review requested at {gate.value}")
            logger.info(f"Review data: {json.dumps(review_data, indent=2, ensure_ascii=False)}")
            return True  # 自动批准

        from .core.coordinator import QualityGate
        for gate in QualityGate:
            coordinator.register_human_review_callback(gate, human_review_callback)

        # 执行工作流
        result = await coordinator.run(
            research_topic=research_topic,
            year_range=year_range,
            max_results=max_results,
            auto_mode=auto_mode,
        )

        return result.to_dict()

    async def search_only(self, research_topic: str, **kwargs) -> dict:
        """仅执行搜索"""
        await self._ensure_initialized()
        agent = SearchAgent(self.workspace)
        result = await agent.run(research_topic=research_topic, **kwargs)
        return result.to_dict()

    async def screen_only(self, **kwargs) -> dict:
        """仅执行筛选"""
        await self._ensure_initialized()
        agent = ScreenAgent(self.workspace, self.rq_manager)
        result = await agent.run(**kwargs)
        return result.to_dict()

    async def cluster_only(self, **kwargs) -> dict:
        """仅执行聚类"""
        await self._ensure_initialized()
        agent = ClusterAgent(self.workspace)
        result = await agent.run(**kwargs)
        return result.to_dict()

    async def summarize_only(self, **kwargs) -> dict:
        """仅执行摘要"""
        await self._ensure_initialized()
        agent = SummaryAgent(self.workspace)
        result = await agent.run(**kwargs)
        return result.to_dict()

    async def get_status(self) -> dict:
        """获取当前状态"""
        await self._ensure_initialized()
        return {
            "workspace": self.workspace.get_workspace_info(),
            "rq_tree": self.rq_manager.current_tree.to_dict() if self.rq_manager.current_tree else None,
        }

    async def get_literature(self, limit: int = 10) -> list:
        """获取文献列表"""
        await self._ensure_initialized()
        papers = await self.workspace.get_literature()
        return [p.to_dict() for p in papers[:limit]]

    async def get_clusters(self) -> list:
        """获取聚类结果"""
        await self._ensure_initialized()
        clusters = await self.workspace.get_clusters()
        return [c.to_dict() for c in clusters]


async def main_async():
    """异步主函数"""
    parser = argparse.ArgumentParser(
        description="Multi-Agent Literature Review System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 完整流程
  python -m src.cli --topic "deep learning for computer vision" --full

  # 仅搜索
  python -m src.cli --topic "transformer models" --search --max-results 100

  # 查看状态
  python -m src.cli --status
        """
    )

    parser.add_argument(
        "--topic", "-t",
        help="Research topic",
    )
    parser.add_argument(
        "--workspace", "-w",
        default="./workspace",
        help="Workspace directory path",
    )
    parser.add_argument(
        "--year-start", type=int, default=2018,
        help="Start year for literature search",
    )
    parser.add_argument(
        "--year-end", type=int, default=2025,
        help="End year for literature search",
    )
    parser.add_argument(
        "--max-results", type=int, default=500,
        help="Maximum number of results to retrieve",
    )

    # 操作模式
    parser.add_argument(
        "--full", "-f",
        action="store_true",
        help="Run full literature review pipeline",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="Run search only",
    )
    parser.add_argument(
        "--screen",
        action="store_true",
        help="Run screening only",
    )
    parser.add_argument(
        "--cluster",
        action="store_true",
        help="Run clustering only",
    )
    parser.add_argument(
        "--summarize",
        action="store_true",
        help="Run summarization only",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status",
    )
    parser.add_argument(
        "--list-papers",
        action="store_true",
        help="List papers in workspace",
    )
    parser.add_argument(
        "--list-clusters",
        action="store_true",
        help="List clusters in workspace",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        default=True,
        help="Auto mode (skip human reviews)",
    )

    args = parser.parse_args()

    # 创建CLI实例
    cli = LiteratureReviewCLI(workspace_path=args.workspace)

    # 执行相应操作
    if args.status:
        status = await cli.get_status()
        print(json.dumps(status, indent=2, ensure_ascii=False))

    elif args.list_papers:
        papers = await cli.get_literature()
        print(json.dumps(papers, indent=2, ensure_ascii=False))

    elif args.list_clusters:
        clusters = await cli.get_clusters()
        print(json.dumps(clusters, indent=2, ensure_ascii=False))

    elif args.full and args.topic:
        result = await cli.run_review(
            research_topic=args.topic,
            year_range=(args.year_start, args.year_end),
            max_results=args.max_results,
            auto_mode=args.auto,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.search and args.topic:
        result = await cli.search_only(
            research_topic=args.topic,
            year_range=(args.year_start, args.year_end),
            max_results=args.max_results,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.screen:
        result = await cli.screen_only()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.cluster:
        result = await cli.cluster_only()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.summarize:
        result = await cli.summarize_only()
        print(json.dumps(result, indent=2, ensure_ascii=False))

    else:
        parser.print_help()


def main():
    """主入口"""
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
