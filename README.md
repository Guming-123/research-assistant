# Research Assistant - 多Agent文献综述系统

基于多Agent协作的智能文献综述系统。输入研究主题，系统自动完成文献检索、相关性筛选、语义聚类和综述报告生成，输出聚焦**底层原理、公式推导与量化分析**的结构化综述报告。

基于论文 *AI-Augmented Literature Reviews: Efficient Clustering and Summarization for Researchers* (IEEE Access, 2025) 的方法论设计。

## 系统架构

```
用户输入研究主题
       │
       ▼
┌──────────────────────────────────────────────────────┐
│                 Coordinator 协调者                     │
│         任务调度 · 状态管理 · 质量门控                  │
└──────────┬──────────┬──────────┬──────────┬──────────┘
           │          │          │          │
           ▼          ▼          ▼          ▼
     ┌─────────┐┌─────────┐┌─────────┐┌─────────┐
     │ Search  ││ Screen  ││ Cluster ││ Summary │
     │ Agent   ││ Agent   ││ Agent   ││ Agent   │
     └────┬────┘└────┬────┘└────┬────┘└────┬────┘
          │          │          │          │
          ▼          ▼          ▼          ▼
     ┌──────────────────────────────────────────────┐
     │            Shared Workspace                  │
     │       文献库 · 聚类 · 摘要 · 嵌入 · 报告       │
     └──────────────────────────────────────────────┘
```

## 处理流水线

| 阶段 | 功能 | 核心技术 |
|------|------|----------|
| **Search** | 多数据库检索、去重、元数据标准化 | arXiv, PubMed, DBLP, Europe PMC, OpenAlex |
| **Screen** | 论文级余弦相似度 + LLM边界判定 | BGE-small-zh嵌入, CUDA矩阵运算, 三级筛选 |
| **Cluster** | 语义聚类发现研究主题 | GPU-PCA降维 + HDBSCAN, 自适应簇大小 |
| **Summary** | 层级RQ驱动综述生成 | 两段式LLM调用, 反虚构引用提示词 |

### 各阶段详细说明

**Search Agent** — 多源文献检索
- 使用LLM自动生成搜索策略，回退到关键词组合
- 5个学术数据库并行检索（全部免费，无需API Key）
- 基于标题标准化去重

**Screen Agent** — 相关性筛选
- 论文级余弦相似度：将所有论文和RQ问题分别嵌入，计算相似度矩阵
- 三级筛选策略：高分自动通过、低分自动拒绝、边界区间送LLM判定
- 全局LLM信号量(5并发)控制速率

**Cluster Agent** — 语义聚类
- 仅对通过筛选的相关论文进行聚类
- PCA降维（GPU加速）→ t-SNE/PCA到2D → HDBSCAN聚类
- 自适应min_cluster_size：论文少时自动调小
- LLM自动生成簇标签、描述和底层原理共同点

**Summary Agent** — 综述生成
- 按簇提取论文内容（含作者、标题、年份、摘要）
- 并行生成各簇的方法论分析（公式+推导）和应用分析
- 两段式LLM生成完整报告（引言+方法论 / 应用+趋势+结论）
- 始终保存fallback报告，LLM失败时自动降级
- 强制引用格式 `[作者, 年份, 论文标题]`，禁止虚构论文

## 安装

### 前置要求

- Python >= 3.10
- CUDA（可选，GPU加速嵌入计算和PCA降维）

### 安装步骤

```bash
git clone https://github.com/Guming-123/research-assistant.git
cd research_assistant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt

# 或使用 pyproject.toml 安装（开发模式）
pip install -e .

# GPU用户（可选，加速嵌入和降维计算）
pip install -r requirements_gpu.txt
python setup_gpu.py
```

## 配置

### 1. API密钥

```bash
cp .env.example .env
```

编辑 `.env`，填入API密钥：

```bash
# 智谱AI GLM（默认）
OPENAI_API_KEY=your_glm_api_key
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/

# 或 OpenAI
# OPENAI_API_KEY=your_openai_api_key
# OPENAI_BASE_URL=https://api.openai.com/v1
```

### 2. 嵌入模型

系统默认使用本地 BGE-small-zh-v1.5 模型（~100MB），用于论文筛选和聚类阶段的文本嵌入，无需远程API调用。

```bash
# .env 中配置（默认值）
EMBEDDING_MODEL=local-zh          # BGE-small-zh-v1.5, 中文优化, 768维
# EMBEDDING_MODEL=local           # 多语言, ~470MB
# EMBEDDING_MODEL=local-en        # 英文优化, ~90MB
# EMBEDDING_MODEL=embedding-3     # 远程GLM API, 1024维, 按调用收费
```

首次运行自动从 HuggingFace 下载模型到本地缓存。CUDA可用时自动GPU加速。详见 [LOCAL_MODEL_GUIDE.md](LOCAL_MODEL_GUIDE.md)。

### 3. 系统参数

编辑 `config.yaml` 调整各阶段参数：

```yaml
# 核心参数说明
llm:
  default_max_tokens: 8000    # LLM最大输出token

screening:
  nf_threshold: 0.68           # 筛选相似度阈值

clustering:
  min_cluster_size: 20         # HDBSCAN最小簇大小
```

## 使用方法

### Web UI（推荐但运行较慢）

```bash
python run_web.py
```

浏览器打开 `http://localhost:7860`，提供可视化界面：

| Tab | 功能 |
|-----|------|
| New Review | 输入研究主题，执行完整流水线，实时进度 |
| Literature Library | 浏览和筛选已检索的论文 |
| Clustering Visualization | 2D散点图展示论文聚类 |
| Review Reports | 查看生成的综述报告 |
| System Status | 工作空间统计信息和RQ树 |

### CLI 完整流程（较快，但是没有可视化）

```bash
python main.py --topic "subthreshold swing reduction in transistors" --full
```

### CLI 单独执行各阶段

```bash
python main.py --topic "deep learning" --search --max-results 100
python main.py --screen
python main.py --cluster
python main.py --summarize
```

### CLI 查看状态

```bash
python main.py --status
python main.py --list-papers
python main.py --list-clusters
```

### 作为Python模块调用

```python
import asyncio
from src.core import Coordinator, SharedWorkspace, RQManager
from src.agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent

async def main():
    workspace = SharedWorkspace("./workspace")
    rq_manager = RQManager("./workspace")

    coordinator = Coordinator(workspace, rq_manager)
    coordinator.register_agent(SearchAgent(workspace))
    coordinator.register_agent(ScreenAgent(workspace, rq_manager))
    coordinator.register_agent(ClusterAgent(workspace))
    coordinator.register_agent(SummaryAgent(workspace))

    result = await coordinator.run(
        research_topic="deep learning for NLP",
        auto_mode=True,
    )
    print(f"Report: {result.data}")

asyncio.run(main())
```

## 项目结构

```
research_assistant/
├── main.py                        # CLI入口
├── run_web.py                     # Web UI入口 (Gradio)
├── config.yaml                    # 系统配置
├── .env.example                   # 环境变量模板
├── pyproject.toml                 # 项目元数据与依赖
├── requirements.txt               # 核心依赖
├── requirements_gpu.txt           # GPU加速依赖
├── setup_gpu.py                   # GPU环境配置脚本
├── LOCAL_MODEL_GUIDE.md           # 本地嵌入模型指南
├── PROJECT_REPORT.md              # 项目技术报告
├── src/
│   ├── cli.py                     # 命令行界面
│   ├── config/
│   │   └── __init__.py            # 配置加载器 (YAML + 环境变量)
│   ├── core/
│   │   ├── agent.py               # BaseAgent基类 (LLM信号量控制)
│   │   ├── coordinator.py         # 协调者 (流水线编排 + 质量门控)
│   │   ├── workspace.py           # 共享工作区 (JSON持久化)
│   │   └── rq_manager.py          # 研究问题层级管理
│   ├── agents/
│   │   ├── search_agent.py        # 多源检索 + 去重
│   │   ├── screen_agent.py        # 余弦相似度 + LLM三级筛选
│   │   ├── cluster_agent.py       # GPU-PCA + HDBSCAN聚类
│   │   └── summary_agent.py       # 两段式综述生成
│   ├── utils/
│   │   ├── llm.py                 # LLM客户端 (缓存 + 信号量)
│   │   ├── embedding.py           # 嵌入 (BGE本地GPU / 远程API)
│   │   ├── api.py                 # arXiv, Semantic Scholar
│   │   ├── api_extended.py        # PubMed, DBLP, OpenAlex, Europe PMC
│   │   ├── pdf.py                 # PDF文本提取
│   │   ├── text.py                # 文本分块、关键词提取、NF计算
│   │   └── exceptions.py          # 异常层级
│   └── web_ui/
│       ├── app.py                 # Gradio应用定义 (5个Tab)
│       ├── handlers.py            # 异步处理函数
│       └── views.py               # 数据格式化与可视化
├── tests/                         # 测试套件
└── workspace/                     # 运行时数据（自动创建）
    ├── literature/records.json    # 论文数据库
    ├── clusters/results.json      # 聚类结果
    ├── embeddings/embeddings.json # 向量缓存
    └── reports/                   # 生成的综述报告 (.md)
```

## 支持的学术数据库

| 数据库 | 领域 | 需要API Key |
|--------|------|:-----------:|
| arXiv | 计算机、物理、数学 | 否 |
| PubMed | 医学、生物学 | 否 |
| DBLP | 计算机科学 | 否 |
| Europe PMC | 生命科学、生物医学 | 否 |
| OpenAlex | 跨学科 | 否 |

> 注意：DBLP和OpenAlex不提供论文摘要，来自这些数据库的论文摘要字段为空。

## 技术栈

| 组件 | 技术 |
|------|------|
| LLM框架 | LangChain + OpenAI兼容接口 |
| 默认LLM | 智谱AI GLM-4-flash/plus |
| 文本嵌入 | BGE-small-zh-v1.5 (本地GPU推理) |
| 降维 | PCA (PyTorch CUDA) |
| 聚类 | HDBSCAN |
| 向量索引 | FAISS (GPU > CPU > NumPy回退) |
| Web UI | Gradio |
| 异步框架 | asyncio + aiohttp |
| 并发控制 | asyncio.Semaphore(5) |

## 报告质量保障

系统通过多层提示词工程确保综述质量，强制聚焦底层原理和公式推导：

- **强制公式解读**：每个方法必须列出核心公式，逐符号解释含义、量纲和典型取值范围，并说明公式的推导来源（哪个物理定律或数学定理）
- **推导链条要求**：从基本假设到最终结果的完整推导路径，识别关键近似步骤
- **理论极限推导**：从第一性原理（热力学、量子力学、信息论等）推导理论上限，并与实验值对比
- **原理层级对比**：方法间的对比必须在公式层面解释差异（哪个公式项导致了性能差异），禁止"A优于B"无理由表述
- **反虚构引用**：强制格式 `[作者, 年份, 论文标题]`，禁止引用列表外的论文，附正确/错误示例
- **反空泛表述**：禁止"取得了较好效果"、"有广阔前景"类空洞评价，必须给出具体指标和数值
- **双重保障**：始终生成fallback报告，LLM失败时自动降级

## 运行示例

```bash
# 完整流程
$ python main.py --topic "methods for reducing subthreshold swing of transistors" --full

# 输出示例:
# workspace/reports/literature_review_20260515_163000.md         <- LLM综述报告
# workspace/reports/literature_review_20260515_163000_fallback.md <- 模板报告(备用)
```

## 测试

```bash
# 运行全部测试
pytest

# 运行单个测试文件
pytest tests/test_workspace.py -v

# 查看覆盖率
pytest --cov=src --cov-report=term-missing
```

## 许可证

MIT License
