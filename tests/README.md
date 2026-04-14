# 测试指南 - Multi-Agent Literature Review System

## 快速开始

### 运行所有测试

```bash
# 使用测试脚本（推荐）
bash run_tests.sh

# 或直接使用pytest
pytest tests/ -v
```

### 运行特定测试文件

```bash
# 测试工作区
pytest tests/test_workspace.py -v

# 测试筛选Agent
pytest tests/test_screen_agent.py -v

# 测试RQ管理器
pytest tests/test_rq_manager.py -v

# 测试配置系统
pytest tests/test_config.py -v
```

### 生成覆盖率报告

```bash
# 生成HTML覆盖率报告
pytest tests/ --cov=src --cov-report=html

# 查看报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

## 测试说明

### 核心测试覆盖

| 测试文件 | 覆盖模块 | 测试用例数 |
|---------|---------|-----------|
| test_workspace.py | SharedWorkspace | 10+ |
| test_screen_agent.py | ScreenAgent | 10+ |
| test_rq_manager.py | RQManager | 12+ |
| test_config.py | Config系统 | 10+ |

### 关键测试场景

#### 1. chunk_id 处理
- ✅ 安全分隔符解析
- ✅ 带下划线的paper_id处理
- ✅ 边界情况处理

#### 2. NF计算
- ✅ 使用实际chunk数计算
- ✅ 零chunk避免除零
- ✅ 边界值验证

#### 3. 两阶段筛选
- ✅ 高置信度直接通过 (NF >= 0.8)
- ✅ 低置信度直接拒绝 (NF < 0.5)
- ✅ 边界案例LLM判定 (0.5 <= NF < 0.8)

#### 4. 配置管理
- ✅ YAML配置加载
- ✅ 环境变量覆盖
- ✅ 默认值回退

#### 5. 并发安全
- ✅ 双重检查锁模式
- ✅ 防止重复初始化

## 持续集成

### GitHub Actions 配置示例

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - run: pip install -r requirements.txt
      - run: pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3
```

## 问题排查

### 测试失败处理

1. **ImportError**: 确保所有依赖已安装
   ```bash
   pip install -r requirements.txt
   ```

2. **AsyncIO警告**: 确保使用 `pytest-asyncio`
   ```bash
   pip install pytest-asyncio
   ```

3. **配置文件未找到**: 测试会自动创建临时配置

## 下一步

- [ ] 添加 Search/Cluster/Summary Agent 测试
- [ ] 添加端到端集成测试
- [ ] 提高覆盖率到 80%+