# 测试文档

本项目使用 pytest 进行单元测试和集成测试。

## 安装测试依赖

```bash
pip install pytest pytest-asyncio pytest-cov
```

## 运行测试

### 运行所有测试
```bash
pytest
```

### 运行特定测试文件
```bash
pytest tests/test_workspace.py
pytest tests/test_agents.py
pytest tests/test_rq_manager.py
pytest tests/test_integration.py
```

### 运行特定测试类或方法
```bash
pytest tests/test_workspace.py::TestSharedWorkspace::test_add_literature
```

### 查看详细输出
```bash
pytest -v
```

### 显示打印输出
```bash
pytest -s
```

### 运行测试并生成覆盖率报告
```bash
pytest --cov=src --cov-report=html
```

## 测试结构

```
tests/
├── conftest.py              # pytest配置和fixtures
├── test_workspace.py        # SharedWorkspace测试
├── test_rq_manager.py       # RQManager测试
├── test_agents.py           # 各Agent测试
└── test_integration.py      # 集成测试
```

## 测试覆盖范围

### test_workspace.py
- LiteratureRecord 创建和转换
- 文献添加、获取、更新、删除
- 聚类保存和加载
- 摘要保存和获取
- Embedding保存和获取
- 检查点创建和恢复

### test_rq_manager.py
- ResearchQuestion 创建和转换
- RQ树初始化
- 各级别问题获取
- 问题添加
- 保存和加载
- 导出报告

### test_agents.py
- SearchAgent 测试
- ScreenAgent 测试
- ClusterAgent 测试
- SummaryAgent 测试
- 输入验证
- 查询构建
- 文档分块
- 降维

### test_integration.py
- 完整搜索工作流
- 聚类工作流
- RQ树工作流
- 工作区持久化
- 检查点工作流

## Fixtures

测试使用以下fixtures（在 `conftest.py` 中定义）：

- `event_loop`: 异步事件循环
- `temp_workspace`: 临时工作区目录
- `workspace`: SharedWorkspace 实例
- `sample_paper`: 单个示例论文
- `sample_papers`: 多个示例论文

## 编写新测试

1. 在对应的测试文件中添加测试方法
2. 使用 `@pytest.mark.asyncio` 装饰器标记异步测试
3. 使用 fixtures 获取测试数据
4. 使用 assert 语句验证结果

示例：

```python
@pytest.mark.asyncio
async def test_my_feature(workspace, sample_paper):
    """测试我的新功能"""
    # 准备
    record = LiteratureRecord(...)
    await workspace.add_literature(record)

    # 执行
    result = await workspace.get_literature()

    # 验证
    assert len(result) == 1
    assert result[0].id == record.id
```

## CI/CD

测试可以在 CI/CD 流水线中自动运行：

```yaml
- name: Run tests
  run: pytest --cov=src --cov-report=xml
```
