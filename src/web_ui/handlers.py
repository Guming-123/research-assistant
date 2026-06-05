"""
异步处理函数 —— 连接 UI 事件与后端 pipeline
"""

import asyncio
import json
import logging
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..core import Coordinator, SharedWorkspace, RQManager
from ..agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent
from ..core.coordinator import QualityGate, Stage
from ..core.rq_manager import RQLevel
from .views import (
    format_cluster_details,
    format_cluster_papers,
    format_cluster_scatter,
    format_papers_dataframe,
    format_report_list,
    render_progress_timeline,
    render_rq_tree_html,
)

logger = logging.getLogger(__name__)

# 模块级状态：记录流水线运行状态防止并发
_pipeline_running = False


class _LogHandler(logging.Handler):
    """将 Python logging 输出追加到列表缓冲区，供 UI 实时读取

    性能优化：
    - 只格式化 INFO 及以上级别的消息（DEBUG 直接跳过）
    - 缓冲区限制最大 500 条，超出后丢弃旧条目
    - progress_callback 限频调用，避免高频开销
    """

    MAX_BUFFER_SIZE = 500

    def __init__(self, buffer: list, progress_callback=None, get_stage_info=None):
        super().__init__()
        self.buffer = buffer
        self.progress_callback = progress_callback
        self.get_stage_info = get_stage_info
        self._last_cb_time = 0.0

    def emit(self, record):
        # 跳过 DEBUG 级别，减少无用的格式化开销
        if record.levelno < logging.INFO:
            return

        msg = self.format(record)

        # 有界缓冲区：超出上限时丢弃最旧的 20%
        if len(self.buffer) >= self.MAX_BUFFER_SIZE:
            del self.buffer[: self.MAX_BUFFER_SIZE // 5]

        self.buffer.append(msg)

        # progress_callback 限频：最多每 3 秒调用一次
        if self.progress_callback and self.get_stage_info:
            import time as _t
            now = _t.monotonic()
            if now - self._last_cb_time >= 3.0:
                self._last_cb_time = now
                cur, done = self.get_stage_info()
                self.progress_callback(cur, done, msg)


def _fmt_duration(seconds: float) -> str:
    """将秒数格式化为可读时长"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


class UIState:
    """UI 会话状态，管理 workspace 实例"""

    def __init__(self, workspace_path: str = "./workspace"):
        self.workspace_path = workspace_path
        self.workspace = SharedWorkspace(workspace_path)
        self.rq_manager = RQManager(workspace_path)
        self._initialized = False

    async def ensure_initialized(self):
        if self._initialized:
            return
        try:
            await self.workspace.load_all()
            await self.rq_manager.load()
            # 从 RQ 树恢复最近的 topic
            if not self.workspace.research_topic and self.rq_manager.current_tree:
                self.workspace.set_topic(self.rq_manager.current_tree.research_topic)
        except Exception:
            pass
        self._initialized = True


# ──────────────────────────────────────────────
# 流水线执行
# ──────────────────────────────────────────────

async def run_pipeline(
    state: UIState,
    topic: str,
    year_start: int,
    year_end: int,
    max_results: int,
    auto_mode: bool,
    progress_callback=None,
    logs_buffer: Optional[list] = None,
) -> Tuple[str, str]:
    """
    运行完整文献综述流程。

    Args:
        logs_buffer: 外部日志缓冲区，用于实时流式输出。传入后 pipeline
                     会将日志追加到此列表，调用方可轮询读取。

    Returns:
        (进度时间线 HTML, 日志文本)
    """
    global _pipeline_running
    if _pipeline_running:
        return render_progress_timeline("initialization", []), "已有任务正在运行，请等待完成。"

    if not topic or not topic.strip():
        return render_progress_timeline("initialization", []), "请输入研究主题。"

    _pipeline_running = True
    logs = logs_buffer if logs_buffer is not None else []
    buf_handler = None

    try:
        await state.ensure_initialized()

        workspace = state.workspace
        rq_manager = state.rq_manager

        # 创建 Coordinator
        coordinator = Coordinator(workspace=workspace, rq_manager=rq_manager)
        coordinator.register_agent(SearchAgent(workspace))
        coordinator.register_agent(ScreenAgent(workspace, rq_manager))
        coordinator.register_agent(ClusterAgent(workspace))
        coordinator.register_agent(SummaryAgent(workspace))

        # 自动批准所有质量门
        async def auto_approve(gate, review_data):
            return True

        for gate in QualityGate:
            coordinator.register_human_review_callback(gate, auto_approve)

        # 注册 BufferHandler：捕获所有 src.* 的日志到 logs_buffer
        src_logger = logging.getLogger("src")
        src_logger.setLevel(logging.INFO)  # 关键：确保 INFO 级别日志被处理

        def get_stage_info():
            return (
                coordinator.task_state.current_stage.value,
                [s.value for s in coordinator.task_state.completed_stages],
            )

        buf_handler = _LogHandler(logs, progress_callback, get_stage_info)
        buf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        src_logger.addHandler(buf_handler)

        import time as _time
        start_time = _time.monotonic()

        result = await coordinator.run(
            research_topic=topic.strip(),
            year_range=(year_start, year_end),
            max_results=max_results,
            auto_mode=auto_mode,
        )

        total_elapsed = _time.monotonic() - start_time
        cur = coordinator.task_state.current_stage.value
        done = [s.value for s in coordinator.task_state.completed_stages]

        # 从 metrics 中提取各阶段耗时
        stage_durations = {}
        metrics = coordinator.task_state.metrics
        for stage_val in done:
            key = f"{stage_val}_metrics"
            if key in metrics and isinstance(metrics[key], dict):
                dur = metrics[key].get("elapsed_seconds")
                if dur is not None:
                    stage_durations[stage_val] = _fmt_duration(dur)

        elapsed_str = _fmt_duration(total_elapsed)
        timeline = render_progress_timeline(cur, done, stage_durations, elapsed_str)

        return timeline, "\n".join(logs)

    except Exception as e:
        tb = traceback.format_exc()
        logs.append(f"错误: {e}\n{tb}")
        return render_progress_timeline("initialization", []), "\n".join(logs)
    finally:
        _pipeline_running = False
        if buf_handler:
            logging.getLogger("src").removeHandler(buf_handler)
        _pipeline_running = False


# ──────────────────────────────────────────────
# 单阶段运行
# ──────────────────────────────────────────────

_STAGE_NAMES = {
    "文献搜索": "search",
    "相关度筛选": "screen",
    "语义聚类": "cluster",
    "综述生成": "summary",
}

_STAGE_HTML = {
    "search": "search",
    "screen": "screen",
    "cluster": "cluster",
    "summary": "summary",
}


async def run_single_stage(
    state: UIState,
    stage_label: str,
    topic: str,
    year_start: int,
    year_end: int,
    max_results: int,
    logs_buffer: Optional[list] = None,
    progress_callback=None,
) -> Tuple[str, str]:
    """
    运行单个阶段。

    Args:
        stage_label: UI 中显示的阶段名称（"文献搜索" 等）
        topic: 研究主题
        logs_buffer: 外部日志缓冲区
        progress_callback: 进度回调
    """
    global _pipeline_running
    if _pipeline_running:
        return render_progress_timeline("initialization", []), "已有任务正在运行，请等待完成。"

    _pipeline_running = True
    logs = logs_buffer if logs_buffer is not None else []

    # 注册 BufferHandler
    src_logger = logging.getLogger("src")
    src_logger.setLevel(logging.INFO)  # 关键：确保 INFO 级别日志被处理
    buf_handler = _LogHandler(logs, progress_callback, lambda: (_STAGE_NAMES.get(stage_label, ""), []))
    buf_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    src_logger.addHandler(buf_handler)

    try:
        await state.ensure_initialized()
        workspace = state.workspace
        rq_manager = state.rq_manager

        stage_key = _STAGE_NAMES.get(stage_label, stage_label)
        logs.append(f"开始单阶段运行: {stage_label}")

        import time as _time
        start_time = _time.monotonic()

        if stage_key == "search":
            if not topic or not topic.strip():
                logs.append("搜索需要输入研究主题")
                return render_progress_timeline("initialization", []), "\n".join(logs)
            workspace.set_topic(topic.strip())
            await rq_manager.initialize_from_topic(topic.strip())
            agent = SearchAgent(workspace)
            result = await agent.run(
                research_topic=topic.strip(),
                year_range=(year_start, year_end),
                max_results=max_results,
            )

        elif stage_key == "screen":
            if not rq_manager.current_tree:
                await rq_manager.load()
            if not rq_manager.current_tree:
                logs.append("请先运行搜索阶段以初始化研究问题树")
                return render_progress_timeline("screen", []), "\n".join(logs)
            workspace.set_topic(rq_manager.current_tree.research_topic)
            rq_ids = [rq.id for rq in rq_manager.get_level_questions(RQLevel.LEVEL_1)]
            agent = ScreenAgent(workspace, rq_manager)
            result = await agent.run(rq_ids=rq_ids)

        elif stage_key == "cluster":
            if not workspace.research_topic:
                if rq_manager.current_tree:
                    workspace.set_topic(rq_manager.current_tree.research_topic)
                elif topic and topic.strip():
                    workspace.set_topic(topic.strip())
            import yaml
            cfg_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
            cfg = {}
            if cfg_path.exists():
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            cluster_cfg = cfg.get("agents", {}).get("cluster", {})
            agent = ClusterAgent(workspace)
            result = await agent.run(
                method=cluster_cfg.get("method", "HDBSCAN").upper(),
                min_cluster_size=cluster_cfg.get("min_cluster_size", 6),
                research_topic=workspace.research_topic,
            )

        elif stage_key == "summary":
            if not rq_manager.current_tree:
                await rq_manager.load()
            if not rq_manager.current_tree:
                logs.append("请先运行搜索阶段以初始化研究问题树")
                return render_progress_timeline("summary", []), "\n".join(logs)
            workspace.set_topic(rq_manager.current_tree.research_topic)
            agent = SummaryAgent(workspace)
            result = await agent.run(rq_tree=rq_manager.current_tree)

        else:
            logs.append(f"未知阶段: {stage_label}")
            return render_progress_timeline("initialization", []), "\n".join(logs)

        stage_val = _STAGE_HTML.get(stage_key, stage_key)
        done = [stage_val]
        elapsed_sec = _time.monotonic() - start_time
        stage_durations = {stage_val: _fmt_duration(elapsed_sec)}
        timeline = render_progress_timeline(stage_val, done, stage_durations, _fmt_duration(elapsed_sec))

        if result.errors:
            for e in result.errors:
                logs.append(f"错误: {e}")

        return timeline, "\n".join(logs)

    except Exception as e:
        tb = traceback.format_exc()
        logs.append(f"错误: {e}\n{tb}")
        return render_progress_timeline("initialization", []), "\n".join(logs)
    finally:
        _pipeline_running = False
        src_logger.removeHandler(buf_handler)


# ──────────────────────────────────────────────
# 论文库
# ──────────────────────────────────────────────

async def load_papers(
    state: UIState,
    search_query: str = "",
    source_filter: str = "全部",
    min_relevance: float = 0.0,
) -> pd.DataFrame:
    """加载并筛选论文"""
    await state.ensure_initialized()
    records = await state.workspace.get_literature()
    dicts = [r.to_dict() for r in records]

    # 筛选
    filtered = dicts
    if search_query:
        q = search_query.lower()
        filtered = [
            r for r in filtered
            if q in r.get("title", "").lower()
            or q in r.get("abstract", "").lower()
            or any(q in a.lower() for a in r.get("authors", []))
        ]

    if source_filter and source_filter != "全部":
        filtered = [r for r in filtered if r.get("source") == source_filter]

    if min_relevance > 0:
        filtered = [
            r for r in filtered
            if r.get("relevance_score") is not None and r["relevance_score"] >= min_relevance
        ]

    return format_papers_dataframe(filtered)


async def load_paper_detail(state: UIState, paper_id: str) -> str:
    """获取单篇论文详情"""
    await state.ensure_initialized()
    papers = await state.workspace.get_literature(paper_ids=[paper_id])
    if not papers:
        return "未找到该论文"
    p = papers[0].to_dict()
    authors = ", ".join(p.get("authors", []))
    html = f"""
    <div style="font-family:sans-serif;max-height:400px;overflow-y:auto;">
      <h3>{p.get('title', '')}</h3>
      <p><b>作者:</b> {authors}</p>
      <p><b>年份:</b> {p.get('year', '-')} &nbsp; <b>来源:</b> {p.get('source', '-')}</p>
      {"<p><b>DOI:</b> " + p['doi'] + "</p>" if p.get('doi') else ""}
      {"<p><b>URL:</b> <a href='" + p['url'] + "' target='_blank'>" + p['url'] + "</a></p>" if p.get('url') else ""}
      <p><b>相关度:</b> {f"{p['relevance_score']:.3f}" if p.get('relevance_score') else '-'}</p>
      <hr>
      <p><b>摘要:</b></p>
      <p style="text-align:justify;">{p.get('abstract', '无摘要')}</p>
    </div>
    """
    return html


async def load_sources_list(state: UIState) -> List[str]:
    """获取所有论文来源列表"""
    await state.ensure_initialized()
    records = await state.workspace.get_literature()
    sources = sorted(set(r.source for r in records if r.source))
    return ["全部"] + sources


# ──────────────────────────────────────────────
# 聚类可视化
# ──────────────────────────────────────────────

async def load_cluster_scatter(state: UIState) -> Optional[pd.DataFrame]:
    """加载散点图数据，优先使用 cluster_agent 保存的 t-SNE 坐标"""
    await state.ensure_initialized()

    # 优先使用 cluster_agent 保存的 t-SNE 2D 坐标
    viz_data = await state.workspace.load("cluster_visualization")
    if viz_data and isinstance(viz_data, dict) and "papers" in viz_data:
        papers_viz = viz_data["papers"]
        if papers_viz:
            # 获取 cluster_id → label 映射
            clusters_data = await state.workspace.get_clusters_as_dicts()
            cluster_labels = {-1: "噪声/未分类"}
            for cid_str, cdata in clusters_data.items():
                cid = cdata.get("cluster_id", int(cid_str))
                cluster_labels[int(cid)] = cdata.get("label", f"Cluster {cid}")

            rows = []
            for p in papers_viz:
                cluster_id = p.get("cluster", -1)
                rows.append({
                    "x": p.get("x", 0.0),
                    "y": p.get("y", 0.0),
                    "cluster": cluster_labels.get(cluster_id, f"Cluster {cluster_id}"),
                    "标题": p.get("title", p.get("id", ""))[:60],
                    "论文ID": p.get("id", ""),
                })
            return pd.DataFrame(rows)

    # Fallback: 无保存数据时重新计算 PCA
    clusters_data = await state.workspace.get_clusters_as_dicts()
    lit_data_list = await state.workspace.get_literature_as_dicts()
    lit_data = {r["id"]: r for r in lit_data_list}
    emb_data = await state.workspace.get_embeddings_as_dict()

    return format_cluster_scatter(clusters_data, lit_data, emb_data)


async def load_cluster_overview(state: UIState) -> pd.DataFrame:
    """聚类概览表"""
    await state.ensure_initialized()
    clusters_data = await state.workspace.get_clusters_as_dicts()
    return format_cluster_details(clusters_data)


async def load_cluster_detail(state: UIState, cluster_id: int) -> Tuple[str, pd.DataFrame]:
    """加载单个聚类的描述和论文列表"""
    await state.ensure_initialized()
    cluster = await state.workspace.get_cluster(cluster_id)
    if not cluster:
        return "未找到该聚类", pd.DataFrame()

    cdict = cluster.to_dict()
    desc = f"**{cdict['label']}**\n\n{cdict['description']}\n\n"
    desc += f"**论文数:** {cdict['size']}  \n"
    desc += f"**子主题:** {', '.join(cdict.get('sub_themes', []))}\n\n"
    desc += "**代表性论文:**\n"
    for rp in cdict.get("representative_papers", []):
        desc += f"- {rp.get('title', '')}  \n  {rp.get('reason', '')}\n"

    # 论文列表
    lit_data_list = await state.workspace.get_literature_as_dicts()
    lit_data = {r["id"]: r for r in lit_data_list}
    df = format_cluster_papers(cdict, lit_data)

    return desc, df


# ──────────────────────────────────────────────
# 综述报告
# ──────────────────────────────────────────────

def load_report_list(state: UIState) -> List[Tuple[str, str]]:
    """获取报告文件列表"""
    reports_dir = Path(state.workspace_path) / "reports"
    return format_report_list(str(reports_dir))


def load_report_content(report_path: str) -> str:
    """读取报告 Markdown 内容"""
    p = Path(report_path)
    if not p.exists():
        return "报告文件不存在"
    return p.read_text(encoding="utf-8")


def find_latest_report(state: UIState) -> Optional[str]:
    """
    查找最新生成的综述报告（优先主报告，回退到 fallback）。

    Returns:
        报告文件的完整路径，找不到则返回 None
    """
    reports_dir = Path(state.workspace_path) / "reports"
    if not reports_dir.exists():
        return None

    reports = sorted(reports_dir.glob("*.md"), reverse=True)
    # 优先返回非 fallback 报告
    main_reports = [r for r in reports if "_fallback" not in r.name]
    if main_reports:
        return str(main_reports[0])
    if reports:
        return str(reports[0])
    return None


# ──────────────────────────────────────────────
# 研究问题树
# ──────────────────────────────────────────────

def load_rq_tree_html(state: UIState) -> str:
    """加载 RQ 树并渲染为 HTML"""
    rq_path = Path(state.workspace_path) / "rq_tree.json"
    if not rq_path.exists():
        return "<p>暂无研究问题数据。请先运行一次综述流程。</p>"
    data = json.loads(rq_path.read_text(encoding="utf-8"))
    if not data:
        return "<p>暂无研究问题数据。请先运行一次综述流程。</p>"
    return render_rq_tree_html(data)


# ──────────────────────────────────────────────
# 系统状态
# ──────────────────────────────────────────────

async def load_workspace_status(state: UIState) -> Dict[str, Any]:
    """获取工作空间统计"""
    await state.ensure_initialized()
    info = state.workspace.get_workspace_info()

    records = await state.workspace.get_literature()
    clusters = await state.workspace.get_clusters()
    reports_dir = Path(state.workspace_path) / "reports"
    report_count = len(list(reports_dir.glob("*.md"))) if reports_dir.exists() else 0

    return {
        "论文总数": len(records),
        "聚类数": len(clusters),
        "报告数": report_count,
        "工作空间路径": str(Path(state.workspace_path).resolve()),
        **info,
    }


async def load_checkpoints(state: UIState) -> pd.DataFrame:
    """获取检查点列表"""
    cp_dir = Path(state.workspace_path) / "checkpoints"
    if not cp_dir.exists():
        return pd.DataFrame(columns=["名称", "时间"])

    rows = []
    for d in sorted(cp_dir.iterdir(), reverse=True):
        meta_path = d / "metadata.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            rows.append({
                "名称": d.name,
                "时间": meta.get("timestamp", "-"),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["名称", "时间"])


# ──────────────────────────────────────────────
# 多 Topic 管理
# ──────────────────────────────────────────────

def list_topics(state: UIState) -> pd.DataFrame:
    """列出数据库中所有 research_topic"""
    topics = state.workspace.list_topics()
    if not topics:
        return pd.DataFrame(columns=["研究主题", "聚类数", "摘要数"])
    rows = []
    for t in topics:
        rows.append({
            "研究主题": t["research_topic"],
            "聚类数": t["cluster_count"],
            "摘要数": t["summary_count"],
        })
    return pd.DataFrame(rows)


async def select_topic(state: UIState, topic_text: str) -> str:
    """选择一个 topic（通过输入研究主题文本）"""
    if not topic_text or not topic_text.strip():
        return "请输入研究主题"
    tid = state.workspace.set_topic(topic_text.strip())
    # 尝试加载对应的 RQ 树
    try:
        await state.rq_manager.load()
    except Exception:
        pass
    return f"已切换到研究主题: {topic_text.strip()}"
