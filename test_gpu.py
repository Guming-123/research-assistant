#!/usr/bin/env python3
"""
测试 GPU 环境是否正常工作
"""

import sys

print("=== GPU 环境验证 ===\n")

try:
    import torch
    print(f"✓ PyTorch 已安装: {torch.__version__}")

    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"✓ GPU 数量: {torch.cuda.device_count()}")
        print(f"✓ GPU 名称: {torch.cuda.get_device_name(0)}")

        # 检查显存
        props = torch.cuda.get_device_properties(0)
        total_memory_gb = props.total_memory / (1024**3)
        print(f"✓ GPU 显存: {total_memory_gb:.1f} GB")

        print("\n=== GPU 计算测试 ===")

        # 测试 GPU 计算
        import time
        start = time.time()

        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)

        elapsed = time.time() - start
        print(f"✓ GPU 矩阵乘法测试: 通过 ({elapsed:.3f}秒)")
        print(f"✓ 结果形状: {z.shape}")

        print("\n=== GPU 配置状态 ===")
        print("✓ GPU 环境配置成功！")
        print("✓ 可以使用 GPU 加速运行项目")

    else:
        print("✗ CUDA 不可用")
        print("请检查:")
        print("  1. NVIDIA 驱动是否安装")
        print("  2. CUDA 版本是否匹配")
        print("  3. PyTorch 是否安装了 GPU 版本")
        sys.exit(1)

except ImportError as e:
    print(f"✗ PyTorch 未安装: {e}")
    print("\n请先运行: python setup_gpu.py")
    sys.exit(1)
except Exception as e:
    print(f"✗ GPU 测试失败: {e}")
    sys.exit(1)

print("\n下一步:")
print("运行项目: python run_gpu.py --cluster")
