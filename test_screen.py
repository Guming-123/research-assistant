#!/usr/bin/env python3
"""
测试 ScreenAgent 的 LLM 筛选功能
单独运行，验证 JSON 解析和限流修复
"""

import asyncio
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env", override=True)

sys.path.insert(0, str(Path(__file__).parent))

from src.core.workspace import SharedWorkspace
from src.core.rq_manager import RQManager
from src.agents.screen_agent import ScreenAgent
from src.utils.llm import get_llm_client

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test():
    workspace = SharedWorkspace("./workspace")
    await workspace.load_all()

    rq_manager = RQManager("./workspace")
    await rq_manager.load()

    papers = await workspace.get_literature()
    print(f"\n=== Workspace 状态 ===")
    print(f"论文数量: {len(papers)}")
    print(f"Embedding 数量: {len(workspace._embeddings)}")
    print(f"RQ 树: {'有' if rq_manager.current_tree else '无'}")

    if not papers:
        print("没有论文数据，请先运行搜索")
        return

    # 取前 3 篇论文测试
    test_paper_ids = [p.id for p in papers[:3]]
    print(f"\n=== 测试论文 ===")
    for pid in test_paper_ids:
        p = await workspace.get_literature(paper_ids=[pid])
        if p:
            print(f"  {pid}: {p[0].title[:60]}...")

    # 创建 ScreenAgent 并直接测试 LLM 筛选
    from src.config import ScreenConfig
    config = ScreenConfig(
        name="ScreenAgent",
        description="test",
        model="glm-4-flash",
        screening_model="glm-4-flash",
        refinement_model="glm-4-flash",
        temperature=0.3,
    )
    agent = ScreenAgent(workspace, rq_manager, config=config)

    # 获取 RQ 问题
    rq_questions = []
    if rq_manager.current_tree:
        for rq in rq_manager.current_tree.get_all_questions():
            rq_questions.append(rq.question)

    if not rq_questions:
        rq_questions = ["transformer模型在自然语言处理中的应用"]

    print(f"\n=== 测试 Quick Screen (glm-4-flash) ===")
    for pid in test_paper_ids:
        result = await agent._quick_screen(pid, rq_questions)
        if result:
            print(f"  {pid}: rejected (confidence={result.confidence})")
        else:
            print(f"  {pid}: passed quick screen")

    print(f"\n=== 测试 Refinement (glm-4-plus) ===")
    for pid in test_paper_ids:
        result = await agent._refine_screening(pid, rq_questions, nf_score=0.8)
        print(f"  {pid}: relevant={result.relevant}, confidence={result.confidence:.2f}")
        print(f"    reasoning: {result.reasoning[:80]}...")

    print(f"\n=== 测试完成 ===")


if __name__ == "__main__":
    asyncio.run(test())
