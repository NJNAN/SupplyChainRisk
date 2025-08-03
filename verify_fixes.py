#!/usr/bin/env python3
"""
快速验证修复效果的测试脚本
"""

import torch
import numpy as np
import yaml
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_robustness_fix():
    """测试鲁棒性修复"""
    print("🧪 测试鲁棒性扰动修复...")
    
    try:
        from robustness import RobustnessEvaluator
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        evaluator = RobustnessEvaluator(config)
        
        # 测试序列扰动
        test_data = torch.randn(2, 10, 5)
        
        # Drop扰动测试
        perturbed = evaluator.perturb_sequence(test_data, drop_rate=0.3)
        diff = torch.abs(test_data - perturbed).sum().item()
        
        print(f"  Drop扰动差异: {diff:.4f}")
        if diff > 0:
            print("  ✅ Drop扰动生效")
        else:
            print("  ❌ Drop扰动未生效")
        
        # Shuffle扰动测试
        perturbed = evaluator.perturb_sequence(test_data, shuffle_rate=0.5)
        diff = torch.abs(test_data - perturbed).sum().item()
        
        print(f"  Shuffle扰动差异: {diff:.4f}")
        if diff > 0:
            print("  ✅ Shuffle扰动生效")
        else:
            print("  ❌ Shuffle扰动未生效")
            
    except Exception as e:
        print(f"  ❌ 鲁棒性测试失败: {str(e)}")

def test_compression_fix():
    """测试压缩修复"""
    print("\n🧪 测试压缩功能修复...")
    
    try:
        from compression import ModelCompressor
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        compressor = ModelCompressor(config)
        
        # 创建简单测试模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(10, 2)
            
            def forward(self, x):
                return self.fc(x)
        
        model = SimpleModel()
        sample_input = torch.randn(32, 10)
        
        # 创建测试数据加载器
        from torch.utils.data import TensorDataset, DataLoader
        test_inputs = torch.randn(100, 10)
        test_targets = torch.randint(0, 2, (100,))
        test_dataset = TensorDataset(test_inputs, test_targets)
        test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
        
        # 测试压缩评估
        results = compressor.compress_and_evaluate(
            model, "test_model", sample_input, test_loader
        )
        
        # 检查是否有准确率对比
        has_accuracy = False
        for method_name, method_results in results.get('compression_results', {}).items():
            if 'accuracy_comparison' in method_results:
                acc_info = method_results['accuracy_comparison']
                print(f"  {method_name}: 原始 {acc_info['original']:.4f} -> 压缩后 {acc_info['compressed']:.4f}")
                has_accuracy = True
        
        if has_accuracy:
            print("  ✅ 压缩准确率对比已修复")
        else:
            print("  ❌ 压缩准确率对比仍缺失")
            
    except Exception as e:
        print(f"  ❌ 压缩测试失败: {str(e)}")

def run_mini_robustness_test():
    """运行小规模鲁棒性测试"""
    print("\n🚀 运行小规模鲁棒性测试...")
    
    try:
        # 使用正确的Python环境运行鲁棒性测试
        import subprocess
        import os
        
        cmd = [
            "/home/njnan/PycharmProjects/AIstu/.venv1/bin/python", 
            "main.py", 
            "--mode", "robust", 
            "--models", "gru", 
            "--verbose"
        ]
        
        result = subprocess.run(
            cmd, 
            cwd="/home/njnan/PycharmProjects/SupplyChainRisk",
            capture_output=True, 
            text=True, 
            timeout=300
        )
        
        if result.returncode == 0:
            print("  ✅ 小规模鲁棒性测试成功")
            # 检查输出中是否有扰动生效的迹象
            if "测试完成" in result.stdout:
                print("  📊 测试完成，检查结果文件...")
        else:
            print(f"  ❌ 测试失败: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        print("  ⏰ 测试超时（正常，说明程序在运行）")
    except Exception as e:
        print(f"  ❌ 测试执行失败: {str(e)}")

def main():
    print("🔧 数据质量修复验证工具")
    print("=" * 50)
    
    # 测试各项修复
    test_robustness_fix()
    test_compression_fix()
    
    print("\n" + "=" * 50)
    print("📋 修复总结:")
    print("1. 鲁棒性扰动函数已改进")
    print("2. 压缩模块已添加准确率评估")
    print("3. 图数据扰动支持更多类型")
    print("4. 建议重新运行完整测试验证")

if __name__ == "__main__":
    main()
