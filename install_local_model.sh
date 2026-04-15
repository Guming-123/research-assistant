#!/bin/bash
# 本地 Embedding 模型快速安装脚本

echo "================================"
echo "本地 Embedding 模型安装"
echo "================================"
echo ""

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ 未找到 Python，请先安装 Python 3.8+"
    exit 1
fi

echo "✅ Python 版本: $(python --version)"
echo ""

# 询问用户选择
echo "请选择要安装的模型："
echo "1) BGE 中文小模型 (推荐，~100MB)"
echo "2) 多语言模型 (~470MB)"
echo "3) 英文小模型 (~90MB)"
echo "4) 完整安装 (所有依赖)"
echo ""
read -p "请输入选项 (1-4): " choice

case $choice in
    1)
        echo ""
        echo "安装 BGE 中文模型..."
        pip install FlagEmbedding
        echo ""
        echo "✅ 安装完成！"
        echo "请在 .env 文件中设置: EMBEDDING_MODEL=local-zh"
        ;;
    2)
        echo ""
        echo "安装多语言模型..."
        pip install sentence-transformers
        echo ""
        echo "✅ 安装完成！"
        echo "请在 .env 文件中设置: EMBEDDING_MODEL=local"
        ;;
    3)
        echo ""
        echo "安装英文小模型..."
        pip install sentence-transformers
        echo ""
        echo "✅ 安装完成！"
        echo "请在 .env 文件中设置: EMBEDDING_MODEL=local-en"
        ;;
    4)
        echo ""
        echo "安装完整依赖..."
        pip install sentence-transformers FlagEmbedding
        echo ""
        echo "✅ 安装完成！"
        echo "可用模型: local-zh, local, local-en, bge-small-zh-v1.5, m3e-base"
        ;;
    *)
        echo "❌ 无效选项"
        exit 1
        ;;
esac

echo ""
echo "================================"
echo "下一步："
echo "1. 确认 .env 文件中设置了 EMBEDDING_MODEL"
echo "2. 运行程序，首次运行会自动下载模型"
echo "3. 模型会缓存到本地，以后无需重新下载"
echo "================================"
