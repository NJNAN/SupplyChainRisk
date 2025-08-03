#!/usr/bin/env python3
"""
测试设备一致性修复的脚本
"""

import torch
import torch.nn as nn
from torch_geometric.data import Data, DataLoader
import numpy as np
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robustness import RobustnessEvaluator
from src.models.gnn_models import GATModel, GCNModel, GraphSAGEModel
import yaml

def create_test_graph_data(batch_size=4, num_nodes=10, num_features=93):
    """创建测试图数据"""
    graphs = []
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for _ in range(batch_size):
        # 节点特征
        x = torch.randn(num_nodes, num_features, device=device)
        
        # 边索引 - 创建一个简单的连通图
        edge_index = []
        for i in range(num_nodes - 1):
            edge_index.extend([[i, i+1], [i+1, i]])  # 双向边
        edge_index = torch.tensor(edge_index, dtype=torch.long, device=device).t()
        
        # 边特征
        edge_attr = torch.randn(edge_index.size(1), 3, device=device)
        
        # 图标签
        y = torch.randint(0, 2, (1,), device=device)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(graph)
    
    return graphs

def test_device_consistency():
    """测试设备一致性"""
    print("🧪 测试设备一致性修复...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 创建鲁棒性评估器
    evaluator = RobustnessEvaluator(config)
    
    # 创建测试图数据
    test_graphs = create_test_graph_data(batch_size=8)
    test_loader = DataLoader(test_graphs, batch_size=2, shuffle=False)
    
    # 创建简单的GNN模型用于测试
    model = GATModel(
        input_dim=93,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        num_classes=2,
        heads=4
    ).to(device)
    
    print("✅ 测试图扰动...")
    try:
        # 测试图扰动
        for batch in test_loader:
            batch = batch.to(device)
            print(f"原始批次设备: x={batch.x.device}, edge_index={batch.edge_index.device}")
            
            # 测试节点丢弃
            perturbed_batch = evaluator.perturb_graph(batch, drop_rate=0.2)
            print(f"扰动后设备: x={perturbed_batch.x.device}, edge_index={perturbed_batch.edge_index.device}")
            
            # 测试模型前向传播
            model.eval()
            with torch.no_grad():
                outputs = model(perturbed_batch)
                print(f"模型输出设备: {outputs.device}")
            
            break  # 只测试第一个批次
        
        print("✅ 图扰动测试通过！")
    except Exception as e:
        print(f"❌ 图扰动测试失败: {str(e)}")
        return False
    
    print("✅ 测试噪声添加...")
    try:
        # 创建序列数据测试噪声添加
        seq_data = torch.randn(4, 24, 93, device=device)
        noisy_data = evaluator.add_noise(seq_data, noise_level=0.1)
        print(f"原始数据设备: {seq_data.device}")
        print(f"噪声数据设备: {noisy_data.device}")
        
        print("✅ 噪声添加测试通过！")
    except Exception as e:
        print(f"❌ 噪声添加测试失败: {str(e)}")
        return False
    
    print("🎉 所有设备一致性测试通过！")
    return True

def test_robustness_evaluation():
    """测试鲁棒性评估"""
    print("🧪 测试鲁棒性评估...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        evaluator = RobustnessEvaluator(config)
        
        # 创建测试数据
        test_graphs = create_test_graph_data(batch_size=16)
        test_loader = DataLoader(test_graphs, batch_size=4, shuffle=False)
        
        # 创建模型
        model = GraphSAGEModel(
            input_dim=93,
            hidden_dim=64,
            num_layers=2,
            dropout=0.1,
            num_classes=2
        ).to(device)
        
        # 测试鲁棒性评估
        results = evaluator.evaluate_model_robustness(model, test_loader, 'graphsage', 'drop')
        
        print(f"基准准确率: {results['baseline_accuracy']:.4f}")
        print(f"基准F1分数: {results['baseline_f1']:.4f}")
        print(f"鲁棒性曲线数量: {len(results['robustness_curves'])}")
        
        print("✅ 鲁棒性评估测试通过！")
        return True
        
    except Exception as e:
        print(f"❌ 鲁棒性评估测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🔧 开始设备一致性修复测试...")
    print("=" * 60)
    
    # 测试设备一致性
    success1 = test_device_consistency()
    
    print("\n" + "=" * 60)
    
    # 测试鲁棒性评估
    success2 = test_robustness_evaluation()
    
    print("\n" + "=" * 60)
    
    if success1 and success2:
        print("🎉 所有测试通过！设备一致性问题已修复。")
        exit(0)
    else:
        print("❌ 部分测试失败，需要进一步检查。")
        exit(1)
