#!/usr/bin/env python3
"""
数据质量和鲁棒性问题修复脚本
"""

import torch
import numpy as np
import json
import yaml
from pathlib import Path

def check_robustness_data():
    """检查鲁棒性测试数据"""
    print("🔍 检查鲁棒性测试数据...")
    
    results_file = Path("results/robustness_results.json")
    if not results_file.exists():
        print("❌ 鲁棒性结果文件不存在")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n📊 鲁棒性测试结果分析:")
    
    for model_key, model_data in data.items():
        if not model_key.endswith('_drop'):
            continue
            
        model_name = model_data['model_type']
        baseline_acc = model_data['baseline_accuracy']
        curves = model_data['robustness_curves']
        
        print(f"\n{model_name.upper()}:")
        print(f"  基准准确率: {baseline_acc:.4f}")
        
        # 检查是否所有扰动级别的结果都相同
        accuracies = [curves[level]['accuracy'] for level in curves.keys()]
        if len(set(accuracies)) == 1:
            print(f"  ⚠️  问题：所有扰动级别准确率相同 ({accuracies[0]:.4f})")
        else:
            print(f"  ✅ 扰动生效，准确率范围: {min(accuracies):.4f} - {max(accuracies):.4f}")

def check_compression_data():
    """检查压缩测试数据"""
    print("\n🔍 检查压缩测试数据...")
    
    results_file = Path("results/compression_results.json")
    if not results_file.exists():
        print("❌ 压缩结果文件不存在")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n📊 压缩测试结果分析:")
    
    for model_name, model_data in data.items():
        print(f"\n{model_name.upper()}:")
        
        compression_summary = model_data.get('compression_summary', {})
        for method, metrics in compression_summary.items():
            print(f"  {method}:")
            print(f"    压缩比: {metrics.get('compression_ratio', 'N/A')}")
            print(f"    参数减少: {metrics.get('parameter_reduction', 'N/A')}")
            print(f"    速度提升: {metrics.get('speedup', 'N/A')}")
            
            # 检查是否有准确率对比
            if 'accuracy_comparison' not in model_data:
                print(f"    ⚠️  缺少准确率对比数据")

def check_model_performance():
    """检查模型性能数据"""
    print("\n🔍 检查模型性能数据...")
    
    results_file = Path("results/evaluation_results.json")
    if not results_file.exists():
        print("❌ 评估结果文件不存在")
        return
    
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n📊 模型性能分析:")
    
    for model_name, metrics in data.items():
        print(f"\n{model_name.upper()}:")
        
        auc_mean = metrics.get('auc', {}).get('mean', 0)
        accuracy_mean = metrics.get('accuracy', {}).get('mean', 0)
        
        print(f"  准确率: {accuracy_mean:.4f}")
        print(f"  AUC: {auc_mean:.4f}")
        
        if auc_mean < 0.6:
            print(f"  ⚠️  AUC过低，可能存在数据不平衡或模型问题")
        elif auc_mean > 0.8:
            print(f"  ✅ AUC良好")
        else:
            print(f"  ⚖️  AUC中等")

def create_robustness_fix():
    """创建鲁棒性修复脚本"""
    print("\n🔧 生成鲁棒性修复建议...")
    
    fix_script = """
# 鲁棒性问题修复建议

## 问题1: 扰动没有真正生效
原因: drop_rate=0时直接返回原数据，但测试中很多level实际上都被设为0

解决方案:
1. 确保扰动参数正确传递
2. 在扰动函数中添加调试信息
3. 验证扰动后数据确实发生了变化

## 问题2: 图数据扰动的批次处理
原因: 批次中的图结构扰动可能没有正确更新

解决方案:
1. 确保图的edge_index正确更新
2. 验证节点特征矩阵的维度匹配
3. 正确处理batch信息

## 问题3: 序列数据扰动策略
当前策略: 用相邻节点填充被drop的位置
建议改进: 使用mask机制或随机填充
"""
    
    print(fix_script)

def suggest_compression_fixes():
    """压缩问题修复建议"""
    print("\n🔧 压缩问题修复建议...")
    
    suggestions = [
        "1. 在压缩后重新评估模型性能，计算准确率下降",
        "2. 使用压缩前后的checkpoint进行对比测试",
        "3. 添加压缩后的模型验证流程",
        "4. 量化功能在CPU上测试，确保功能正常",
        "5. 记录压缩过程中的详细日志"
    ]
    
    for suggestion in suggestions:
        print(f"  {suggestion}")

def main():
    print("🚀 数据质量检查和修复建议工具")
    print("=" * 50)
    
    # 检查各种数据
    check_model_performance()
    check_robustness_data()
    check_compression_data()
    
    # 提供修复建议
    create_robustness_fix()
    suggest_compression_fixes()
    
    print("\n" + "=" * 50)
    print("🎯 主要问题总结:")
    print("1. 鲁棒性测试中扰动没有真正生效")
    print("2. 压缩后缺少性能重新评估")
    print("3. 部分模型AUC偏低，需要检查数据平衡性")
    print("4. 需要添加更多的数据验证和日志记录")

if __name__ == "__main__":
    main()
