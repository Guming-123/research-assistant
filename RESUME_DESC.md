# 简历项目描述

## 中文版

**多Agent自动化文献综述系统** | Python, LangChain, PyTorch(CUDA), HDBSCAN, asyncio

基于 LangChain 构建多Agent协作的文献综述自动化系统，采用 Coordinator-Agent 编排模式管理四阶段异步流水线。SearchAgent 通过 LLM 自动生成领域限定查询，从 arXiv/PubMed/DBLP/OpenAlex 等多源学术数据库检索并去重；ScreenAgent 利用 BGE/all-MiniLM 本地嵌入模型计算论文与层级研究问题的余弦相似度，结合 LLM 边界判定实现三级筛选；ClusterAgent 通过 GPU 加速 PCA 降维 + HDBSCAN 聚类自动发现研究主题；SummaryAgent 采用两段式 LLM 调用生成结构化综述报告，通过反幻觉提示词工程强制精确引用格式。全局 asyncio 信号量控制 API 并发，共享工作区实现跨Agent数据持久化与检查点恢复，单次运行可在数分钟内完成数百篇文献的系统性综述。
