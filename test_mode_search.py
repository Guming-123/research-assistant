#!/usr/bin/env python3
"""
Test Mode - 使用模拟数据测试系统（无需 API 调用）
"""

import asyncio
import sys
from pathlib import Path

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).parent))

from src.core import SharedWorkspace, RQManager
from src.agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent
from src.core.workspace import LiteratureRecord
from mock_search import MOCK_PAPERS


async def test_mode():
    """测试模式：使用模拟数据"""
    print("=" * 60)
    print("Multi-Agent Literature Review System - Test Mode")
    print("=" * 60)
    print("\n📦 使用模拟数据，无需 API 调用\n")

    # 初始化工作区
    workspace = SharedWorkspace("./workspace_test")
    await workspace.load_all()

    # 清空现有数据
    print("🗑️  清空现有数据...")
    await workspace._literature.clear()
    await workspace._persist()

    # 添加模拟论文数据
    print(f"📚 添加 {len(MOCK_PAPERS)} 篇模拟论文...")
    records = []
    for paper in MOCK_PAPERS:
        record = LiteratureRecord(
            id=paper["paperId"],
            title=paper["title"],
            authors=[a["name"] for a in paper["authors"]],
            abstract=paper["abstract"],
            year=paper["year"],
            source=paper["source"],
            url=paper["url"],
            venue=paper.get("venue"),
            citation_count=paper.get("citationCount"),
        )
        records.append(record)

    await workspace.add_literature(records)
    print(f"✅ 已添加 {len(records)} 篇论文到工作区")

    # 初始化 RQ 管理器
    rq_manager = RQManager("./workspace_test")
    await rq_manager.initialize_from_topic("deep learning in computer vision")

    print("\n📊 工作区状态:")
    info = await workspace.get_workspace_info()
    print(f"  - 文献数量: {info['literature_count']}")
    print(f"  - 簇数量: {info['cluster_count']}")

    print("\n" + "=" * 60)
    print("✅ 测试模式初始化完成！")
    print("=" * 60)
    print("\n现在可以运行各阶段：")
    print("  python main.py --screen")
    print("  python main.py --cluster")
    print("  python main.py --summarize")
    print(f"\n工作区路径: {workspace.base_path}")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(test_mode())
