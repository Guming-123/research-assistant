"""
Research Assistant Web UI 入口
用法: python run_web.py
启动后访问: http://localhost:7860
"""

import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中
project_root = str(Path(__file__).resolve().parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# 加载环境变量
from dotenv import load_dotenv
load_dotenv()

from src.web_ui.app import create_app


def main():
    workspace = sys.argv[1] if len(sys.argv) > 1 else "./workspace"
    app = create_app(workspace_path=workspace)
    import gradio as gr
    print("\n" + "=" * 50)
    print("  Research Assistant 已启动")
    print("  浏览器访问: http://localhost:7860")
    print("=" * 50 + "\n")
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
        css=".gradio-container { max-width: 1200px; margin: auto; } .progress-log textarea { font-family: monospace; font-size: 0.85em; }",
        max_file_size="50mb",
    )
    app.queue(max_size=20)


if __name__ == "__main__":
    main()
