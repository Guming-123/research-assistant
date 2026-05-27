"""
Gradio Web UI 应用定义
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional

import gradio as gr

from .handlers import (
    UIState,
    list_topics,
    load_checkpoints,
    load_cluster_detail,
    load_cluster_overview,
    load_cluster_scatter,
    load_paper_detail,
    load_papers,
    load_report_content,
    load_report_list,
    load_rq_tree_html,
    load_sources_list,
    load_workspace_status,
    run_pipeline,
    run_single_stage,
    select_topic,
)

# 确保 src 包可导入
_project_root = str(Path(__file__).resolve().parent.parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


def _async_run(coro):
    """在已有事件循环内安全运行协程"""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, coro)
            return future.result()
    else:
        return asyncio.run(coro)


def create_app(workspace_path: str = "./workspace") -> gr.Blocks:
    """创建 Gradio 应用"""

    state = UIState(workspace_path)

    css = """
    .gradio-container { max-width: 1200px; margin: auto; }
    .progress-log textarea { font-family: monospace; font-size: 0.85em; }
    """

    with gr.Blocks(title="Research Assistant - 文献综述系统") as app:

        gr.Markdown(
            "# 🔬 Research Assistant\n"
            "多Agent文献综述系统 | 搜索 → 筛选 → 聚类 → 综述"
        )

        # ════════════════════════════════════════
        # Tab 1: 新建综述
        # ════════════════════════════════════════
        with gr.Tab("🚀 新建综述"):
            with gr.Row():
                topic_input = gr.Textbox(
                    label="研究主题",
                    placeholder="例如：deep learning for medical image analysis",
                    lines=2,
                    scale=4,
                )
            with gr.Row():
                year_start = gr.Number(label="起始年份", value=2018, precision=0)
                year_end = gr.Number(label="结束年份", value=2025, precision=0)
                max_results = gr.Number(label="最大结果数", value=500, precision=0)
                auto_mode = gr.Checkbox(label="自动模式", value=True)

            stage_selector = gr.Radio(
                choices=["全流程", "文献搜索", "相关度筛选", "语义聚类", "综述生成"],
                value="全流程",
                label="运行阶段",
            )

            run_btn = gr.Button("开始综述", variant="primary", size="lg")
            progress_html = gr.HTML(value="<div>等待开始...</div>")
            log_output = gr.Textbox(
                label="运行日志",
                lines=15,
                max_lines=30,
                interactive=False,
                elem_classes=["progress-log"],
            )

            async def on_run(topic, ys, ye, mr, am, stage, progress=gr.Progress()):
                """运行流水线或单阶段，实时流式输出日志"""
                needs_topic = stage in ("全流程", "文献搜索")
                if needs_topic and (not topic or not topic.strip()):
                    yield "<div>请输入研究主题</div>", "请输入研究主题。"
                    return

                # 限频：progress 回调最多每 2 秒触发一次，减少 SSE 推送开销
                _last_progress_time = [0.0]
                import time as _time

                def progress_cb(cur, done, msg):
                    now = _time.monotonic()
                    if now - _last_progress_time[0] < 2.0:
                        return
                    _last_progress_time[0] = now
                    stage_names = {
                        "initialization": "初始化",
                        "search": "文献搜索",
                        "screen": "相关度筛选",
                        "cluster": "语义聚类",
                        "summary": "综述生成",
                        "completed": "完成",
                    }
                    stages_order = ["initialization", "search", "screen", "cluster", "summary", "completed"]
                    done_count = len(done)
                    frac = (done_count + 0.5) / len(stages_order)
                    progress(frac, desc=f"{stage_names.get(cur, cur)}: {msg[:60]}")

                logs_buffer = []
                last_count = 0

                if stage == "全流程":
                    pipeline_task = asyncio.create_task(
                        run_pipeline(
                            state, topic, int(ys), int(ye), int(mr), am,
                            progress_callback=progress_cb,
                            logs_buffer=logs_buffer,
                        )
                    )
                else:
                    pipeline_task = asyncio.create_task(
                        run_single_stage(
                            state, stage, topic, int(ys), int(ye), int(mr),
                            logs_buffer=logs_buffer,
                            progress_callback=progress_cb,
                        )
                    )

                # 轮询日志缓冲区，有新内容就 yield 更新 UI
                while not pipeline_task.done():
                    await asyncio.sleep(1.0)
                    if len(logs_buffer) > last_count:
                        # 只发送最近 200 行，避免大字符串 join 开销
                        visible = logs_buffer[-200:]
                        yield "<div>运行中...</div>", "\n".join(visible)
                        last_count = len(logs_buffer)

                # 结束，获取最终结果
                try:
                    timeline, final_logs = pipeline_task.result()
                except Exception as e:
                    timeline = "<div>运行出错</div>"
                    final_logs = "\n".join(logs_buffer[-500:]) + f"\n错误: {e}"

                yield timeline, final_logs

            run_btn.click(
                fn=on_run,
                inputs=[topic_input, year_start, year_end, max_results, auto_mode, stage_selector],
                outputs=[progress_html, log_output],
            )

        # ════════════════════════════════════════
        # Tab 2: 论文库
        # ════════════════════════════════════════
        with gr.Tab("📚 论文库"):
            with gr.Row():
                paper_search = gr.Textbox(label="搜索", placeholder="标题/作者/摘要关键词", scale=3)
                paper_source = gr.Dropdown(label="来源", choices=["全部"], value="全部", scale=1)
                paper_relevance = gr.Slider(0, 1, 0, step=0.05, label="最低相关度")
                paper_filter_btn = gr.Button("筛选", scale=0)

            papers_df = gr.Dataframe(
                label="论文列表",
                interactive=False,
                wrap=True,
                column_widths=["5%", "35%", "20%", "8%", "10%", "10%", "8%"],
            )

            paper_detail_html = gr.HTML(label="论文详情", visible=True)

            async def load_papers_data():
                sources = await load_sources_list(state)
                df = await load_papers(state)
                return gr.update(choices=sources, value="全部"), df

            def on_paper_filter(search, source, min_rel):
                return _async_run(load_papers(state, search, source, min_rel))

            def on_paper_select(evt: gr.SelectData):
                if evt.index is None:
                    return ""
                # 获取选中行的第一列（ID 列）
                row_idx = evt.index[0] if isinstance(evt.index, tuple) else evt.index
                # 从当前数据获取 ID
                try:
                    paper_id = evt.value
                except Exception:
                    paper_id = ""
                return _async_run(load_paper_detail(state, str(paper_id)))

            papers_df.select(
                fn=on_paper_select,
                outputs=paper_detail_html,
            )

            paper_filter_btn.click(
                fn=on_paper_filter,
                inputs=[paper_search, paper_source, paper_relevance],
                outputs=papers_df,
            )

            app.load(fn=lambda: _async_run(load_papers_data()), outputs=[paper_source, papers_df])

        # ════════════════════════════════════════
        # Tab 3: 聚类可视化
        # ════════════════════════════════════════
        with gr.Tab("📊 聚类可视化"):
            with gr.Row():
                scatter_plot = gr.ScatterPlot(
                    title="论文聚类 2D 可视化",
                    x="x",
                    y="y",
                    color="cluster",
                    tooltip=["标题", "论文ID"],
                )
                cluster_overview = gr.Dataframe(label="聚类概览", interactive=False)

            with gr.Row():
                cluster_id_input = gr.Number(label="选择聚类 ID", value=0, precision=0)
                cluster_load_btn = gr.Button("加载聚类详情")

            cluster_desc = gr.Markdown(label="聚类描述")
            cluster_papers_df = gr.Dataframe(label="聚类论文", interactive=False)

            def on_load_clusters():
                scatter = _async_run(load_cluster_scatter(state))
                overview = _async_run(load_cluster_overview(state))
                if scatter is None:
                    scatter = gr.update()
                return scatter, overview

            def on_cluster_detail(cid):
                desc, df = _async_run(load_cluster_detail(state, int(cid)))
                return desc, df

            cluster_load_btn.click(
                fn=on_cluster_detail,
                inputs=[cluster_id_input],
                outputs=[cluster_desc, cluster_papers_df],
            )

            app.load(fn=on_load_clusters, outputs=[scatter_plot, cluster_overview])

        # ════════════════════════════════════════
        # Tab 4: 综述报告
        # ════════════════════════════════════════
        with gr.Tab("📄 综述报告"):
            with gr.Row():
                report_select = gr.Dropdown(label="选择报告", choices=[], scale=4)
                report_refresh_btn = gr.Button("刷新列表", scale=0)
                report_download = gr.File(label="下载报告", scale=0)

            report_content = gr.Markdown(label="报告内容", elem_classes=["report-content"])

            def refresh_reports():
                reports = load_report_list(state)
                names = [r[0] for r in reports]
                first_path = reports[0][1] if reports else None
                content = load_report_content(first_path) if first_path else "暂无报告"
                return gr.update(choices=names, value=names[0] if names else None), first_path, content

            def on_report_select(name):
                reports = load_report_list(state)
                for rname, rpath in reports:
                    if rname == name:
                        content = load_report_content(rpath)
                        return rpath, content
                return None, "报告不存在"

            report_refresh_btn.click(
                fn=refresh_reports,
                outputs=[report_select, report_download, report_content],
            )

            report_select.change(
                fn=on_report_select,
                inputs=[report_select],
                outputs=[report_download, report_content],
            )

            app.load(fn=refresh_reports, outputs=[report_select, report_download, report_content])

        # ════════════════════════════════════════
        # Tab 5: 系统状态
        # ════════════════════════════════════════
        with gr.Tab("⚙️ 系统状态"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 工作空间")
                    status_display = gr.JSON(label="当前 Topic 状态")
                with gr.Column(scale=1):
                    gr.Markdown("### 研究问题")
                    rq_display = gr.HTML()

            gr.Markdown("### Topic 管理")
            with gr.Row():
                topics_df = gr.Dataframe(label="所有 Topic", interactive=False)
            with gr.Row():
                topic_input = gr.Textbox(label="输入研究主题切换", placeholder="例如：deep learning for vision", scale=4)
                topic_select_btn = gr.Button("切换 Topic", scale=0)
                topic_msg = gr.Textbox(label="", interactive=False, scale=2)

            gr.Markdown("### 检查点")
            checkpoints_df = gr.Dataframe(label="检查点历史", interactive=False)

            def load_status():
                info = _async_run(load_workspace_status(state))
                rq_html = load_rq_tree_html(state)
                topics = list_topics(state)
                cp_df = _async_run(load_checkpoints(state))
                return info, rq_html, topics, cp_df

            def on_select_topic(text):
                msg = _async_run(select_topic(state, text))
                info = _async_run(load_workspace_status(state))
                rq_html = load_rq_tree_html(state)
                return msg, info, rq_html

            topic_select_btn.click(
                fn=on_select_topic,
                inputs=[topic_input],
                outputs=[topic_msg, status_display, rq_display],
            )

            app.load(fn=load_status, outputs=[status_display, rq_display, topics_df, checkpoints_df])

    return app
