#!/usr/bin/env python3
"""
测试 ClusterAgent 聚类功能
单独验证 numpy.int64 序列化修复
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent))

from src.core.workspace import SharedWorkspace
from src.agents.cluster_agent import ClusterAgent

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test():
    workspace = SharedWorkspace("./workspace")
    await workspace.load_all()

    papers = await workspace.get_literature()
    print(f"\n=== Workspace 状态 ===")
    print(f"论文数量: {len(papers)}")
    print(f"Embedding 数量: {len(workspace._embeddings)}")

    if not papers:
        print("没有论文数据，请先运行搜索")
        return

    agent = ClusterAgent(workspace)
    result = await agent.run(max_papers=100)

    print(f"\n=== 聚类结果 ===")
    print(f"成功: {result.success}")
    print(f"错误: {result.errors}")
    if result.data:
        print(f"簇数量: {result.data.get('cluster_count', 0)}")
        clusters = result.data.get("clusters", [])
        for c in clusters:
            print(f"  簇 {c['cluster_id']}: {c.get('label', 'N/A')} ({c['size']} 篇)")

        # 关键测试：JSON 序列化
        try:
            json_str = json.dumps(result.data, ensure_ascii=False)
            print(f"\nJSON 序列化: 成功 ({len(json_str)} bytes)")
        except TypeError as e:
            print(f"\nJSON 序列化: 失败 - {e}")

    print(f"\n=== 测试完成 ===")


if __name__ == "__main__":
    asyncio.run(test())
