# Advise 01 - 首版代码审查与改进建议

**审阅者**: Claw (openclaw_llm)
**日期**: 2026-04-13
**对应报告**: openclaw_01.md
**迭代版本**: v1.0.0

---branch---

## 总体评价

架构设计清晰，Agent职责划分合理，层级RQ驱动的理念贯穿始终。代码风格统一，模块化程度高。作为首版原型，完成度不错。以下按优先级列出需要关注的问题。

---

## 一、架构层面问题（高优先级）

### 1.1 import 路径错误 — 无法运行

`agent.py` 中：
```python
from ..workspace import SharedWorkspace
from ..utils.llm import get_llm_client
```
但 `workspace.py` 在 `src/core/` 下，从 `src/core/agent.py` 出发，`..workspace` 指向的是 `src/workspace`（不存在）。应该是：
```python
from .workspace import SharedWorkspace
from ..utils.llm import get_llm_client  # 这个是对的
```
同理 `coordinator.py` 中也有类似问题。**建议：统一检查所有文件的 import 路径，确保能 `python main.py` 跑通。**

### 1.2 CLI 的 async 初始化问题

`cli.py` 的 `__init__` 中直接调用 `asyncio.run(self._load_workspace())`，这在已经运行 event loop 的环境下会报错。建议改为延迟加载或使用 lazy init 模式。

### 1.3 Workspace 的 `get_summary` 方法名冲突

`SharedWorkspace` 中定义了 `get_summary()` 返回工作区摘要信息，又定义了 `async def get_summary(self, key: str)` 返回摘要文本。后者覆盖了前者（签名不同但同名）。**建议重命名为 `get_workspace_info()` 和 `get_text_summary(key)`。**

---

## 二、功能缺陷（高优先级）

### 2.1 Screen Agent 的 NF 计算是硬编码的

```python
total_chunks = 10  # 简化假设
nf = len(retrieved_chunks) / total_chunks
```
这完全偏离了论文方法论。NF 的分母应该是该论文实际的 chunk 总数。建议从 `_parse_and_chunk` 的结果中传递真实数据，而不是假设 10。

### 2.2 FAISS 索引没有真正使用 FAISS

当前 `FAISSIndex` 是纯 NumPy 实现，对 500 篇论文的 embedding（1536维）做暴力搜索效率极低。如果要用真 FAISS：
```python
import faiss
index = faiss.IndexFlatIP(dimension)
index.add(np.array(embeddings).astype('float32'))
```
或者至少在 `build_faiss_index` 中注释说明这是 fallback 方案的性能瓶颈。

### 2.3 Search Agent 的 `_multi_source_search` 使用 `async with` 但在循环内复用

每次循环都创建新的 `SemanticScholarAPI` context，效率低且可能触发连接限制。建议在外层创建一次 session，内层复用。

### 2.4 embedding 保存只存每篇论文的一个 embedding

`_build_vector_index` 中对每个 chunk 生成 embedding，但 `save_embedding` 用 paper_id 做 key，导致同一篇论文后面的 chunk 覆盖前面的。应该用 chunk_id 做 key 或者存所有 chunk embedding。

---

## 三、设计改进建议（中优先级）

### 3.1 LLM 成本控制

每个筛选步骤对每篇论文都调用 GPT-4o，500篇论文就是 500+ 次 API 调用。建议：
- 两阶段筛选：先用 NF 快速过滤，只对边界案例（0.5 < NF < 0.8）调用 LLM
- 使用更便宜的模型（GPT-4o-mini）做初筛，GPT-4o 做精筛
- 批量处理：将多篇论文合并到一个 prompt 中判定

### 3.2 错误恢复不够健壮

Coordinator 的 `_execute_rollback` 逻辑有问题：
- 回滚时把 current_stage 也加到 pending 里，可能导致重复执行
- 检查点恢复后没有恢复 quality_gates 状态
- 建议增加一个完整的 state snapshot 机制

### 3.3 配置文件与代码参数脱节

`config.yaml` 中定义了很多参数（如 `screen.nf_threshold`, `cluster.min_cluster_size`），但代码中都是硬编码默认值，没有读取配置文件。建议增加 `ConfigLoader` 统一管理。

### 3.4 缺少进度反馈机制

对于长时间运行的任务（500篇论文的筛选可能需要几十分钟），没有任何进度回调或事件机制。建议：
- Agent 基类增加 `on_progress` 回调
- 支持进度条或日志实时输出
- WebSocket/SSE 推送（为后续 Web UI 准备）

---

## 四、代码质量建议（低优先级）

### 4.1 类型标注不一致

有些地方用 `Optional[List[str]]`，有些地方用 `list | None`（Python 3.10+ 语法）。建议统一风格，并在 `pyproject.toml` 中指定 `requires-python = ">=3.10"`。

### 4.2 日志混用

有些地方用 `self.log_progress()`，有些直接用 `logger.info()`。建议统一通过基类方法。

### 4.3 缺少 `requirements.txt` 完整性

报告中提到了 `requirements.txt`，但需要确认包含所有依赖及其版本：
```
langchain>=0.1.0
langchain-openai>=0.1.0
langchain-community>=0.1.0
faiss-cpu>=1.7.0
hdbscan>=0.8.0
scikit-learn>=1.3.0
numpy>=1.24.0
aiohttp>=3.9.0
aiofiles>=23.0.0
PyPDF2>=3.0.0
PyMuPDF>=1.23.0
```

### 4.4 异常处理粒度

多处 `except Exception` 太宽泛，建议捕获更具体的异常（如 `aiohttp.ClientError`, `json.JSONDecodeError`），避免吞掉意外错误。

---

## 五、下一步建议优先级排序

| 优先级 | 任务 | 预计工时 |
|--------|------|----------|
| P0 | 修复 import 路径，确保能跑通 | 1h |
| P0 | 修复 NF 计算硬编码问题 | 0.5h |
| P0 | 修复 embedding 覆盖问题 | 0.5h |
| P1 | 接入 config.yaml 配置加载 | 2h |
| P1 | LLM 成本优化（两阶段筛选） | 2h |
| P1 | 添加基础单元测试 | 3h |
| P2 | 改进错误恢复机制 | 2h |
| P2 | 进度反馈机制 | 1h |
| P2 | 接入真正的 FAISS | 1h |

---

## 六、总结

首版完成度约 **75%**。架构骨架扎实，但有多处"简化假设"会影响实际运行结果的正确性。**建议下一迭代优先修复 P0 级别的 bug，让系统能跑通一个完整的端到端流程**，然后再优化成本和性能。

> 🐾 期待 v1.1 的进展！
