#!/bin/bash
# 测试运行脚本

echo "================================"
echo "Multi-Agent Literature Review - Tests"
echo "================================"
echo ""

# 检查依赖
echo "检查测试依赖..."
python -c "import pytest" 2>/dev/null || {
    echo "❌ pytest 未安装"
    echo "请运行: pip install pytest pytest-asyncio"
    exit 1
}

python -c "import pytest_asyncio" 2>/dev/null || {
    echo "❌ pytest-asyncio 未安装"
    echo "请运行: pip install pytest-asyncio"
    exit 1
}

echo "✅ 依赖检查完成"
echo ""

# 运行测试
echo "运行测试..."
echo ""

pytest -v --tb=short "$@"

# 检查结果
if [ $? -eq 0 ]; then
    echo ""
    echo "================================"
    echo "✅ 所有测试通过！"
    echo "================================"
else
    echo ""
    echo "================================"
    echo "❌ 有测试失败"
    echo "================================"
    exit 1
fi
