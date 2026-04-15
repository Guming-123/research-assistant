# Multi-Agent Literature Review System

基于论文《AI-Augmented Literature Reviews: Efficient Clustering and Summarization for Researchers》(IEEE Access, 2025) 设计的智能文献综述系统。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    Multi-Agent 文献综述系统                       │
│                                                                 │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │ Search   │──▶│ Screen   │──▶│ Cluster  │──▶│ Summary  │    │
│  │ Agent    │   │ Agent    │   │ Agent    │   │ Agent    │    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘    │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Coordinator Agent (协调者)                   │   │
│  └──────────────────────────────────────────────────────────┘   │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Shared Workspace (共享工作区)                 │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## 核心功能

1. **Search Agent**: 多源文献检索、去重、标准化
   - **支持的数据库**：arXiv、PubMed、DBLP、Europe PMC、OpenAlex（全部免费）
   - 无需 API Key，开箱即用
2. **Screen Agent**: 基于归一化频率(NF)的相关性筛选
3. **Cluster Agent**: HDBSCAN语义聚类、主题发现
4. **Summary Agent**: 层级RQ驱动的结构化摘要生成
5. **Coordinator**: 全局协调、质量门控、状态管理

## 安装

```bash
# 克隆仓库
git clone https://github.com/openclaw-llm/research-assistant.git
cd research_assistant

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

## 配置

1. 复制环境变量模板：
```bash
cp .env.example .env
```

2. 编辑`.env`文件，填入你的API密钥：

### 使用 GLM API (智谱AI) - 推荐

```bash
# 获取 API Key: https://open.bigmodel.cn/usercenter/apikeys
OPENAI_API_KEY=your_glm_api_key_here
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_MODEL=glm-4-plus
```

**GLM 模型选择**：
- `glm-4-flash`: 快速模型，适合大规模调用（成本较低）
- `glm-4-plus`: 标准模型，适合通用任务
- `glm-4`: 基础模型
- `glm-4-long`: 长上下文模型 (128K tokens)
- `embedding-2` / `embedding-3`: Embedding 模型

### 使用 OpenAI API

```bash
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o
```

## 使用方法

### 完整流程

```bash
python main.py --topic "deep learning in computer vision" --full
```

### 单独执行各阶段

```bash
# 仅搜索
python main.py --topic "transformer models" --search --max-results 100

# 仅筛选
python main.py --screen

# 仅聚类
python main.py --cluster

# 仅生成摘要
python main.py --summarize
```

### 查看状态

```bash
# 查看系统状态
python main.py --status

# 列出已检索的论文
python main.py --list-papers

# 列出聚类结果
python main.py --list-clusters
```

## 作为Python模块使用

```python
import asyncio
from src.core import Coordinator, SharedWorkspace, RQManager
from src.agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent

async def main():
    # 初始化
    workspace = SharedWorkspace("./workspace")
    rq_manager = RQManager("./workspace")

    # 创建协调者
    coordinator = Coordinator(workspace, rq_manager)

    # 注册Agent
    coordinator.register_agent(SearchAgent(workspace))
    coordinator.register_agent(ScreenAgent(workspace, rq_manager))
    coordinator.register_agent(ClusterAgent(workspace))
    coordinator.register_agent(SummaryAgent(workspace))

    # 执行
    result = await coordinator.run(
        research_topic="deep learning for NLP",
        year_range=(2018, 2025),
        auto_mode=True,
    )

    print(result)

asyncio.run(main())
```

## 项目结构

```
research_assistant/
├── src/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                 # 命令行界面
│   ├── core/                  # 核心组件
│   │   ├── agent.py           # Agent基类
│   │   ├── coordinator.py     # 协调者
│   │   ├── workspace.py       # 共享工作区
│   │   └── rq_manager.py      # RQ管理器
│   ├── agents/                # 专业Agent
│   │   ├── search_agent.py
│   │   ├── screen_agent.py
│   │   ├── cluster_agent.py
│   │   └── summary_agent.py
│   └── utils/                 # 工具模块
│       ├── llm.py
│       ├── pdf.py
│       ├── text.py
│       ├── embedding.py
│       └── api.py
├── main.py                    # 主入口
├── config.yaml                # 配置文件
├── requirements.txt           # 依赖
└── README.md
```

## 技术栈

- **LLM框架**: LangChain (支持 GLM API)
- **向量检索**: FAISS / NumPy fallback
- **聚类**: HDBSCAN / scikit-learn
- **PDF处理**: PyMuPDF / PyPDF2
- **异步IO**: asyncio / aiohttp

### 支持的论文数据库（全部免费）

| 数据库 | 领域 | 说明 |
|--------|------|------|
| **arXiv** | 计算机科学、物理、数学 | 预印本论文，更新快 |
| **PubMed** | 医学、生物学 | 美国国家医学图书馆 |
| **DBLP** | 计算机科学 | 计算机科学文献数据库 |
| **Europe PMC** | 生命科学、生物医学 | 欧洲开放获取文献 |
| **OpenAlex** | 跨学科 | 最大的开放学术引用数据库 |

### LLM 提供商支持

系统通过 LangChain 的 OpenAI 兼容接口支持多种 LLM 提供商：

| 提供商 | 模型示例 | BASE_URL |
|--------|----------|----------|
| **智谱AI (GLM)** | glm-4-plus, glm-4-flash | `https://open.bigmodel.cn/api/paas/v4/` |
| **OpenAI** | gpt-4o, gpt-4o-mini | `https://api.openai.com/v1` |
| **其他兼容接口** | - | 根据服务商配置 |

## 引用

如果本系统对您的研究有帮助，请引用原论文：

```bibtex
@article{ai_augmented_literature_reviews_2025,
  title={AI-Augmented Literature Reviews: Efficient Clustering and Summarization for Researchers},
  journal={IEEE Access},
  year={2025},
  note{基于此论文的方法论设计}
}
```

## 许可证

MIT License
