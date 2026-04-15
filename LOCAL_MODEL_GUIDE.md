# 本地 Embedding 模型使用指南

## 快速开始

### 1. 安装依赖

根据您的需求选择安装：

```bash
# 方案1: 最小安装（英文为主，90MB）
pip install sentence-transformers

# 方案2: 中文优化安装（推荐，100MB）
pip install FlagEmbedding

# 方案3: 完整安装（支持所有模型）
pip install sentence-transformers FlagEmbedding
```

### 2. 模型选择

在 `.env` 文件中设置 `EMBEDDING_MODEL`：

```bash
# 中文优化（推荐）
EMBEDDING_MODEL=local-zh

# 多语言
EMBEDDING_MODEL=local

# 英文优化
EMBEDDING_MODEL=local-en
```

### 3. 首次运行

程序会在第一次运行时自动下载模型（约100-500MB），请确保网络畅通。

---

## 可用模型列表

| 模型名称 | 描述 | 大小 | 语言 | 推荐场景 |
|---------|------|------|------|----------|
| `local-zh` | BGE中文小模型 | ~100MB | 中文 | **推荐**，中文文献 |
| `local` | 多语言模型 | ~470MB | 多语言 | 中英混合文献 |
| `local-en` | 英文小模型 | ~90MB | 英文 | 英文文献 |
| `bge-small-zh-v1.5` | BGE中文小模型 | ~100MB | 中文 | 中文文献 |
| `bge-base-zh-v1.5` | BGE中文基础模型 | ~400MB | 中文 | 中文文献（更高精度） |
| `m3e-base` | M3E中文模型 | ~400MB | 中文 | 中文文献 |

---

## 配置示例

### .env 文件

```bash
# 本地 embedding 模型
EMBEDDING_MODEL=local-zh

# LLM 模型（如果需要）
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=glm-4-plus
```

---

## 使用示例

### Python 代码

```python
from src.utils.embedding import get_embeddings

# 使用本地模型
texts = ["这是一篇关于深度学习的论文", "机器学习是人工智能的分支"]
embeddings = await get_embeddings(texts, model="local-zh")
print(f"Generated {len(embeddings)} embeddings")
```

### 命令行

```bash
# 运行完整流程
python main.py --topic "深度学习在计算机视觉中的应用" --full

# 只运行聚类
python main.py --cluster
```

---

## 性能对比

| 模型 | 速度 | 内存占用 | 准确度 | 成本 |
|-----|------|---------|--------|------|
| `local-zh` (BGE) | 快 | ~2GB | 高 | 免费 |
| `local` (多语言) | 中等 | ~3GB | 中高 | 免费 |
| `embedding-3` (GLM API) | 快 | 低 | 高 | 按调用收费 |

---

## 常见问题

### Q1: 模型下载失败？
A: 检查网络连接，或手动下载模型到本地缓存目录：
```bash
# Linux/Mac: ~/.cache/huggingface/
# Windows: C:\Users\<用户名>\.cache\huggingface\
```

### Q2: 内存不足？
A: 使用更小的模型：
```bash
EMBEDDING_MODEL=local-en  # 90MB
```

### Q3: 想切换回 API？
A: 修改 `.env` 文件：
```bash
EMBEDDING_MODEL=embedding-3  # 使用 GLM API
```

---

## 推荐配置

### 中文文献综述
```bash
EMBEDDING_MODEL=local-zh
OPENAI_MODEL=glm-4-plus
```

### 英文文献综述
```bash
EMBEDDING_MODEL=local-en
OPENAI_MODEL=gpt-4o
```

### 中英混合文献
```bash
EMBEDDING_MODEL=local
OPENAI_MODEL=glm-4-plus
```
