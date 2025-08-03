#!/usr/bin/env python3
"""
鲁棒性测试扰动修复脚本
"""

import torch
import numpy as np
import random
from torch.utils.data import DataLoader, TensorDataset
import yaml

def test_sequence_perturbation():
    """测试序列扰动功能"""
    print("🔍 测试序列数据扰动...")
    
    # 创建测试数据
    batch_size, seq_len, features = 4, 10, 5
    test_data = torch.randn(batch_size, seq_len, features)
    
    print(f"  原始数据shape: {test_data.shape}")
    print(f"  原始数据前2个样本的第一个时间步:")
    print(f"    样本0: {test_data[0, 0, :3]}")
    print(f"    样本1: {test_data[1, 0, :3]}")
    
    # 测试drop扰动
    drop_rate = 0.3
    perturbed_data = perturb_sequence_fixed(test_data, drop_rate=drop_rate)
    
    print(f"\n  Drop扰动后 (drop_rate={drop_rate}):")
    print(f"    样本0: {perturbed_data[0, 0, :3]}")
    print(f"    样本1: {perturbed_data[1, 0, :3]}")
    
    # 检查是否有变化
    diff = torch.abs(test_data - perturbed_data).sum()
    print(f"  数据差异总和: {diff.item():.4f}")
    
    if diff.item() > 0:
        print("  ✅ 扰动生效")
    else:
        print("  ❌ 扰动未生效")

def perturb_sequence_fixed(data: torch.Tensor, drop_rate: float = 0.0, shuffle_rate: float = 0.0) -> torch.Tensor:
    """修复后的序列扰动函数"""
    if drop_rate == 0.0 and shuffle_rate == 0.0:
        return data

    perturbed_data = data.clone()
    batch_size, seq_len, features = data.shape

    for batch_idx in range(batch_size):
        sequence = perturbed_data[batch_idx]  # [seq_len, features]

        # 节点丢弃 - 改进策略
        if drop_rate > 0:
            # 生成drop mask
            drop_mask = torch.rand(seq_len, device=data.device) < drop_rate
            num_dropped = drop_mask.sum().item()
            
            if num_dropped > 0 and num_dropped < seq_len:  # 确保不会全部丢弃
                # 方法1: 用零填充
                sequence[drop_mask] = 0
                
                # 方法2: 用噪声填充 (注释掉，可选择使用)
                # noise = torch.randn_like(sequence[drop_mask]) * 0.1
                # sequence[drop_mask] = noise

        # 顺序打乱 - 改进策略
        if shuffle_rate > 0:
            num_shuffles = max(1, int(seq_len * shuffle_rate))
            
            # 随机交换位置
            for _ in range(num_shuffles):
                idx1, idx2 = random.sample(range(seq_len), 2)
                sequence[idx1], sequence[idx2] = sequence[idx2].clone(), sequence[idx1].clone()

        perturbed_data[batch_idx] = sequence

    return perturbed_data

def test_graph_perturbation():
    """测试图扰动功能"""
    print("\n🔍 测试图数据扰动...")
    
    from torch_geometric.data import Data
    
    # 创建测试图
    num_nodes = 10
    x = torch.randn(num_nodes, 16)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    
    graph = Data(x=x, edge_index=edge_index)
    print(f"  原始图: {num_nodes}个节点, {edge_index.size(1)}条边")
    print(f"  原始节点特征 (前3个): {x[:3, :3]}")
    
    # 测试节点丢弃
    drop_rate = 0.3
    perturbed_graph = perturb_graph_fixed(graph, drop_rate=drop_rate)
    
    print(f"\n  Drop扰动后 (drop_rate={drop_rate}):")
    print(f"    节点数: {perturbed_graph.x.size(0)}")
    print(f"    边数: {perturbed_graph.edge_index.size(1)}")
    
    if perturbed_graph.x.size(0) < num_nodes:
        print("  ✅ 图扰动生效")
    else:
        print("  ❌ 图扰动未生效")

def perturb_graph_fixed(graph: 'Data', drop_rate: float = 0.0) -> 'Data':
    """修复后的图扰动函数"""
    if drop_rate == 0.0:
        return graph

    perturbed_graph = graph.clone()
    num_nodes = graph.x.size(0)

    # 节点丢弃
    if drop_rate > 0 and num_nodes > 1:
        keep_nodes = max(1, int(num_nodes * (1 - drop_rate)))
        
        # 随机选择要保留的节点
        keep_indices = torch.randperm(num_nodes)[:keep_nodes]
        keep_indices = torch.sort(keep_indices)[0]

        # 更新节点特征
        perturbed_graph.x = graph.x[keep_indices]
        
        # 更新边索引
        if hasattr(graph, 'edge_index') and graph.edge_index.size(1) > 0:
            # 创建节点映射
            node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}
            
            # 过滤边：只保留两个端点都在保留节点中的边
            old_edge_index = graph.edge_index
            valid_edges = []
            
            for i in range(old_edge_index.size(1)):
                src, dst = old_edge_index[0, i].item(), old_edge_index[1, i].item()
                if src in node_mapping and dst in node_mapping:
                    valid_edges.append([node_mapping[src], node_mapping[dst]])
            
            if valid_edges:
                perturbed_graph.edge_index = torch.tensor(valid_edges, dtype=torch.long).t()
            else:
                # 如果没有有效边，创建自环
                perturbed_graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)

    return perturbed_graph

def create_robustness_fix_patch():
    """生成鲁棒性修复补丁"""
    print("\n🔧 生成鲁棒性修复补丁...")
    
    patch_content = '''
# 鲁棒性模块修复补丁

## 修复要点:

1. 序列扰动函数改进:
   - drop操作改为直接置零而不是用相邻值填充
   - shuffle操作改为真正的随机交换
   - 添加设备一致性检查

2. 图扰动函数改进:
   - 正确更新边索引和节点映射
   - 确保扰动后图结构的合法性
   - 处理边界情况

3. 扰动效果验证:
   - 添加扰动前后的数据对比
   - 记录扰动参数和实际效果
   - 确保不同扰动类型都能正确应用
'''
    
    print(patch_content)

def main():
    print("🚀 鲁棒性测试扰动修复工具")
    print("=" * 50)
    
    # 测试扰动功能
    test_sequence_perturbation()
    test_graph_perturbation()
    
    # 生成修复建议
    create_robustness_fix_patch()
    
    print("\n" + "=" * 50)
    print("🎯 修复方案:")
    print("1. 替换robustness.py中的扰动函数")
    print("2. 添加扰动验证日志")
    print("3. 确保所有扰动类型都正确实现")

if __name__ == "__main__":
    main()
