"""
数据格式转换与渲染辅助函数
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def format_papers_dataframe(records: List[Dict[str, Any]]) -> pd.DataFrame:
    """将文献记录列表转为展示用 DataFrame"""
    if not records:
        return pd.DataFrame(columns=["标题", "作者", "年份", "来源", "相关度"])

    rows = []
    for r in records:
        authors = r.get("authors", [])
        author_str = ", ".join(authors[:3])
        if len(authors) > 3:
            author_str += " et al."

        score = r.get("relevance_score")
        score_str = f"{score:.2f}" if score is not None else "-"

        rows.append({
            "ID": r.get("id", ""),
            "标题": r.get("title", "")[:100],
            "作者": author_str,
            "年份": r.get("year", "-"),
            "来源": r.get("source", "-"),
            "相关度": score_str,
            "聚类": r.get("cluster_id", "-"),
        })

    return pd.DataFrame(rows)


def format_cluster_scatter(
    clusters_data: Dict[str, Dict],
    literature_data: Dict[str, Dict],
    embeddings_data: Dict[str, List[float]],
) -> Optional[pd.DataFrame]:
    """
    从 clusters + literature + embeddings 构建散点图数据。

    使用 embeddings 的前两个 PCA 分量作为 x, y 近似坐标。
    如果 embeddings 不可用则返回 None。
    """
    if not embeddings_data or not clusters_data:
        return None

    import numpy as np
    from sklearn.decomposition import PCA

    # 收集所有有 embedding 的论文
    paper_ids = list(embeddings_data.keys())
    if not paper_ids:
        return None

    emb_matrix = np.array([embeddings_data[pid] for pid in paper_ids])
    if emb_matrix.ndim != 2 or emb_matrix.shape[1] < 2:
        return None

    # PCA 降到 2 维
    n_components = min(2, emb_matrix.shape[1], emb_matrix.shape[0])
    pca = PCA(n_components=n_components)
    coords_2d = pca.fit_transform(emb_matrix)

    # 构建 paper_id -> cluster_id 映射
    paper_to_cluster: Dict[str, int] = {}
    cluster_labels: Dict[int, str] = {}
    for cid_str, cdata in clusters_data.items():
        cid = cdata.get("cluster_id", int(cid_str))
        cluster_labels[cid] = cdata.get("label", f"Cluster {cid}")
        for pid in cdata.get("paper_ids", []):
            paper_to_cluster[pid] = cid

    rows = []
    for i, pid in enumerate(paper_ids):
        lit = literature_data.get(pid, {})
        rows.append({
            "x": float(coords_2d[i, 0]),
            "y": float(coords_2d[i, 1]),
            "cluster": cluster_labels.get(
                paper_to_cluster.get(pid, -1), "未分类"
            ),
            "标题": lit.get("title", pid)[:60],
            "论文ID": pid,
        })

    return pd.DataFrame(rows)


def render_rq_tree_html(rq_tree_dict: Dict[str, Any]) -> str:
    """将 RQ 树转为可读的 HTML 嵌套列表"""
    topic = rq_tree_dict.get("research_topic", "")
    roots = rq_tree_dict.get("root_questions", [])

    level_colors = {1: "#2e7d32", 2: "#1565c0", 3: "#e65100"}
    level_labels = {1: "维度", 2: "子问题", 3: "细节"}

    def render_node(node: Dict, level: int) -> str:
        qid = node.get("id", "")
        question = node.get("question", "")
        desc = node.get("description", "")
        status = node.get("status", "pending")
        status_icon = "✅" if status == "completed" else "⏳"
        color = level_colors.get(level, "#333")
        label = level_labels.get(level, "")

        children_html = ""
        children = node.get("children", [])
        if children:
            child_items = "\n".join(render_node(c, level + 1) for c in children)
            children_html = f"<ul>{child_items}</ul>"

        return (
            f'<li style="margin:4px 0;">'
            f'<span style="color:{color};font-weight:bold;">[{qid}]</span> '
            f'<span style="font-size:0.85em;color:#888;">{status_icon} {label}</span><br>'
            f'<span style="margin-left:20px;">{question}</span>'
            f'{children_html}'
            f'</li>'
        )

    items = "\n".join(render_node(r, 1) for r in roots)
    return (
        f'<div style="font-family:sans-serif;">'
        f'<h4 style="margin-bottom:8px;">研究主题：{topic}</h4>'
        f'<ul style="list-style:none;padding-left:0;">{items}</ul>'
        f'</div>'
    )


def render_progress_timeline(
    current_stage: str,
    completed_stages: List[str],
    stage_durations: Optional[Dict[str, str]] = None,
    elapsed: str = "",
) -> str:
    """生成流水线阶段进度的 HTML 时间线（含耗时）"""
    stages = [
        ("initialization", "初始化"),
        ("search", "文献搜索"),
        ("screen", "相关度筛选"),
        ("cluster", "语义聚类"),
        ("summary", "综述生成"),
        ("completed", "完成"),
    ]
    durations = stage_durations or {}

    items = []
    for stage_key, stage_name in stages:
        dur = durations.get(stage_key)
        dur_text = f" <span style='font-size:0.75em;color:#888;'>({dur})</span>" if dur else ""

        if stage_key in completed_stages:
            color = "#2e7d32"
            icon = "●"
            weight = "bold"
        elif stage_key == current_stage:
            color = "#1565c0"
            icon = "▶"
            weight = "bold"
        else:
            color = "#bbb"
            icon = "○"
            weight = "normal"

        items.append(
            f'<span style="color:{color};font-weight:{weight};font-size:1.1em;">'
            f'{icon} {stage_name}{dur_text}</span>'
        )

    elapsed_html = ""
    if elapsed:
        elapsed_html = f'<span style="font-size:0.9em;color:#666;margin-left:12px;">总耗时: {elapsed}</span>'

    return '<div style="display:flex;gap:16px;flex-wrap:wrap;align-items:center;">' + "  →  ".join(items) + elapsed_html + "</div>"


def format_report_list(reports_dir: str) -> List[Tuple[str, str]]:
    """扫描报告目录，返回 (显示名, 文件路径) 列表"""
    p = Path(reports_dir)
    if not p.exists():
        return []

    reports = sorted(p.glob("*.md"), reverse=True)
    return [(f.name, str(f)) for f in reports]


def format_cluster_details(clusters_data: Dict[str, Dict]) -> pd.DataFrame:
    """将聚类结果转为 DataFrame 概览"""
    rows = []
    for cid_str, cdata in clusters_data.items():
        rows.append({
            "聚类ID": cdata.get("cluster_id", cid_str),
            "标签": cdata.get("label", "-"),
            "论文数": cdata.get("size", 0),
            "子主题": ", ".join(cdata.get("sub_themes", [])[:3]),
        })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["聚类ID", "标签", "论文数", "子主题"]
    )


def format_cluster_papers(
    cluster_data: Dict, literature_data: Dict[str, Dict]
) -> pd.DataFrame:
    """返回指定聚类中所有论文的 DataFrame"""
    paper_ids = cluster_data.get("paper_ids", [])
    rows = []
    for pid in paper_ids:
        lit = literature_data.get(pid, {})
        if lit:
            authors = lit.get("authors", [])
            author_str = ", ".join(authors[:2])
            if len(authors) > 2:
                author_str += " et al."
            rows.append({
                "标题": lit.get("title", pid)[:80],
                "作者": author_str,
                "年份": lit.get("year", "-"),
                "来源": lit.get("source", "-"),
                "相关度": f"{lit['relevance_score']:.2f}" if lit.get("relevance_score") else "-",
            })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["标题", "作者", "年份", "来源", "相关度"]
    )
