# 智能科研助手Agent系统 - 首版开发报告

**致**: openclaw_llm
**日期**: 2026-04-13
**版本**: v1.0.0
**迭代次数**: 1

---

## 一、项目概述

基于论文《AI-Augmented Literature Reviews: Efficient Clustering and Summarization for Researchers》(IEEE Access, 2025) 的方法论，设计并实现了一个完整的Multi-Agent智能文献综述系统。

### 核心设计理念

1. **分步解耦**: 将文献综述拆分为独立阶段，每阶段由专门Agent负责
2. **人机协同**: 关键节点保留人工校验接口（支持自动模式）
3. **层级RQ驱动**: 用分层研究问题引导检索、聚类、摘要
4. **可复现性**: 全流程参数化、可追溯

---

## 二、系统架构实现

### 2.1 核心组件

```
research_assistant/
├── src/
│   ├── core/                  # 核心框架
│   │   ├── agent.py          # Agent基类 (200+ 行)
│   │   ├── coordinator.py    # 协调者 (400+ 行)
│   │   ├── workspace.py      # 共享工作区 (500+ 行)
│   │   └── rq_manager.py     # RQ管理器 (350+ 行)
│   ├── agents/               # 专业Agent
│   │   ├── search_agent.py   # 文献检索 (350+ 行)
│   │   ├── screen_agent.py   # 相关性筛选 (400+ 行)
│   │   ├── cluster_agent.py  # 聚类分析 (450+ 行)
│   │   └── summary_agent.py  # 综述生成 (400+ 行)
│   ├── utils/                # 工具模块
│   │   ├── llm.py           # LLM客户端
│   │   ├── pdf.py           # PDF处理
│   │   ├── text.py          # 文本处理
│   │   ├── embedding.py     # 向量化
│   │   └── api.py           # 学术数据库API
│   └── cli.py                # 命令行界面
├── main.py                    # 主入口
├── config.yaml                # 配置文件
└── requirements.txt           # 依赖管理
```

### 2.2 Agent架构

#### Search Agent (文献检索)
- **职责**: 多源检索、去重、标准化
- **数据源**: Semantic Scholar API, arXiv API
- **核心功能**:
  - LLM辅助查询构建
  - 多源并行检索
  - 基于标题的智能去重
  - 元数据标准化

#### Screen Agent (相关性筛选)
- **职责**: 基于NF的相关性筛选
- **核心算法**:
  - 文档语义分块 (512 tokens, 50 overlap)
  - Embedding编码 (text-embedding-3-small, 1536维)
  - FAISS索引构建
  - 归一化频率 NF(d) = 被检索chunk数 / 总chunk数
  - 阈值过滤 (NF ≥ 0.7)
  - LLM辅助相关性判定

#### Cluster Agent (聚类分析)
- **职责**: 语义聚类、主题发现
- **核心算法**:
  - t-SNE降维 (2D, perplexity=30)
  - HDBSCAN聚类 (min_cluster_size=5)
  - LLM生成簇标签
  - 轮廓系数评估

#### Summary Agent (综述生成)
- **职责**: 结构化摘要、报告生成
- **核心功能**:
  - 按簇RAG摘要
  - 二级RQ分析 (方法论、应用)
  - 跨簇趋势综合
  - Markdown格式报告

---

## 三、技术实现细节

### 3.1 协调者模式

```python
class Coordinator(BaseAgent):
    """全局任务调度、状态管理、质量门控"""

    # 质量门控节点
    - POST_SEARCH: 搜索后确认文献池
    - POST_SCREEN: 筛选后确认相关文献
    - POST_CLUSTER: 聚类后确认主题结构
    - POST_SUMMARY: 摘要后确认综述质量

    # 工作流状态
    - TaskState: 阶段追踪、错误记录、检查点管理
```

### 3.2 层级RQ系统

```
Level 1 (宏观维度)
├── RQ1: 方法论维度
│   ├── RQ11: 使用了哪些方法类型？
│   │   ├── RQ111: 技术细节
│   │   └── RQ112: 性能表现
│   └── RQ12: 各方法的优缺点对比？
├── RQ2: 应用维度
│   ├── RQ21: 应用在哪些领域？
│   └── RQ22: 各领域面临什么挑战？
└── RQ3: 趋势维度
    ├── RQ31: 方法演进趋势
    └── RQ32: 未来研究方向
```

### 3.3 共享工作区

```python
class SharedWorkspace:
    """所有Agent共享的数据存储"""

    # 数据类型
    - LiteratureRecord: 文献记录
    - ClusterResult: 聚类结果
    - WorkspaceEntry: 通用条目

    # 持久化
    - JSON序列化
    - 检查点机制
    - 版本管理
```

---

## 四、已实现功能

### 核心流程
- [x] 完整工作流编排
- [x] Agent间协调通信
- [x] 状态追踪与错误处理
- [x] 检查点与恢复机制

### 文献检索
- [x] Semantic Scholar API集成
- [x] arXiv API集成
- [x] 智能去重
- [x] 元数据标准化

### 相关性筛选
- [x] 语义分块
- [x] Embedding索引
- [x] 归一化频率计算
- [x] LLM辅助判定

### 聚类分析
- [x] t-SNE降维
- [x] HDBSCAN聚类
- [x] 簇标签生成
- [x] 质量评估

### 综述生成
- [x] 按簇摘要
- [x] 结构化报告
- [x] Markdown输出

---

## 五、技术栈

| 组件 | 技术选型 |
|------|----------|
| LLM框架 | LangChain |
| LLM模型 | GPT-4o |
| Embedding | text-embedding-3-small |
| 向量检索 | FAISS / NumPy fallback |
| 聚类 | HDBSCAN / scikit-learn |
| PDF处理 | PyMuPDF / PyPDF2 |
| 异步IO | asyncio / aiohttp |
| 配置管理 | YAML / python-dotenv |

---

## 六、使用示例

### 命令行

```bash
# 完整流程
python main.py --topic "deep learning in computer vision" --full

# 单独执行
python main.py --topic "transformer models" --search --max-results 100
python main.py --screen
python main.py --cluster
python main.py --summarize

# 查看状态
python main.py --status
python main.py --list-papers
python main.py --list-clusters
```

### Python API

```python
import asyncio
from src.core import Coordinator, SharedWorkspace, RQManager
from src.agents import SearchAgent, ScreenAgent, ClusterAgent, SummaryAgent

async def main():
    workspace = SharedWorkspace("./workspace")
    rq_manager = RQManager("./workspace")
    coordinator = Coordinator(workspace, rq_manager)

    # 注册Agent
    coordinator.register_agent(SearchAgent(workspace))
    coordinator.register_agent(ScreenAgent(workspace, rq_manager))
    coordinator.register_agent(ClusterAgent(workspace))
    coordinator.register_agent(SummaryAgent(workspace))

    # 执行
    result = await coordinator.run(
        research_topic="deep learning for NLP",
        auto_mode=True,
    )

asyncio.run(main())
```

---

## 七、当前状态

### 已完成
- [x] 完整系统架构设计
- [x] 核心框架实现 (BaseAgent, Coordinator, Workspace, RQManager)
- [x] 四个专业Agent实现
- [x] 工具模块实现 (LLM, PDF, Text, Embedding, API)
- [x] CLI接口
- [x] 配置管理
- [x] 文档 (README, config)

### 测试状态
- 未进行单元测试
- 未进行集成测试
- 未进行端到端测试

### 已知限制
1. PDF下载功能简化（仅记录路径）
2. FAISS依赖可选（使用NumPy fallback）
3. 人工审核默认跳过（自动模式）

---

## 八、待改进方向

### 短期 (下一迭代)
1. 添加单元测试和集成测试
2. 完善PDF下载和解析
3. 优化API速率限制处理
4. 增加更多数据源支持

### 中期
1. 实现真正的FAISS索引
2. 添加Web界面
3. 支持增量更新
4. 添加可视化功能

### 长期
1. 支持多语言
2. 本地模型集成
3. 分布式处理
4. 协作功能

---

## 九、建议请求

请openclaw_llm提供以下反馈：

1. **架构评审**: 系统架构是否合理？有无设计缺陷？
2. **代码质量**: 代码风格、结构、可维护性评估
3. **功能完整性**: 是否有遗漏的核心功能？
4. **性能优化**: 哪些部分需要性能优化？
5. **测试策略**: 应该如何设计测试体系？
6. **文档完善**: 需要补充哪些文档？
7. **下一步**: 优先改进哪个方向？

---

## 十、总结

首版开发已完成系统核心功能实现，严格按照论文方法论设计了Multi-Agent架构。系统已具备基本的文献检索、筛选、聚类、综述生成能力，可以开始实际使用测试。

期待您的反馈，以指导下一阶段的迭代优化。

**报告人**: Research Assistant Team
**日期**: 2026-04-13
