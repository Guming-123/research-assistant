#!/usr/bin/env python3
"""
测试 API 连接性
"""

import asyncio
import aiohttp
import xml.etree.ElementTree as ET


async def test_arxiv():
    """测试 arXiv API"""
    print("Testing arXiv API...")

    base_url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": "all:deep learning",
        "max_results": 5,
        "sortBy": "relevance",
        "sortOrder": "descending"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(base_url, params=params, timeout=aiohttp.ClientTimeout(total=30)) as resp:
                print(f"Status: {resp.status}")
                if resp.status == 200:
                    xml_content = await resp.text()
                    print(f"Response length: {len(xml_content)} bytes")

                    # 解析XML
                    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
                    root = ET.fromstring(xml_content)

                    entries = root.findall("atom:entry", ns)
                    print(f"Found {len(entries)} papers")

                    for entry in entries[:2]:
                        title = entry.find("atom:title", ns).text.strip()
                        print(f"  - {title[:60]}...")
                else:
                    text = await resp.text()
                    print(f"Error response: {text[:500]}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")


async def test_semantic_scholar():
    """测试 Semantic Scholar API"""
    print("\nTesting Semantic Scholar API...")

    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": "deep learning",
        "limit": 5,
        "fields": "paperId,title"
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                print(f"Status: {resp.status}")
                if resp.status == 200:
                    data = await resp.json()
                    total = data.get("total", 0)
                    papers = data.get("data", [])
                    print(f"Total papers: {total}")
                    print(f"Returned papers: {len(papers)}")

                    for paper in papers[:2]:
                        title = paper.get("title", "N/A")
                        print(f"  - {title[:60]}...")
                elif resp.status == 429:
                    print("❌ Rate limited! (429)")
                    print("   Get API key for higher limits: https://www.semanticscholar.org/product/api#api-key")
                else:
                    text = await resp.text()
                    print(f"Error response: {text[:500]}")
    except Exception as e:
        print(f"Exception: {type(e).__name__}: {e}")


async def main():
    print("=" * 60)
    print("API Connectivity Test")
    print("=" * 60)

    await test_arxiv()
    await test_semantic_scholar()

    print("\n" + "=" * 60)
    print("Test complete")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
