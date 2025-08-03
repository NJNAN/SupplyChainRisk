#!/usr/bin/env python3
"""
测试mask和batch修复的简单脚本
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
from src.models.gnn_models import GATModel
import yaml

def test_mask_fix():
    """测试mask形状修复"""
    print("🧪 测试mask形状修复...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建简单的单图测试
    x = torch.randn(20, 93, device=device)  # 20个节点
    edge_index = torch.tensor([[i, (i+1)%20] for i in range(20)] + [[(i+1)%20, i] for i in range(20)], device=device).t()
    edge_attr = torch.randn(edge_index.size(1), 3, device=device)
    y = torch.randint(0, 2, (1,), device=device)
    
    graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    evaluator = RobustnessEvaluator(config)
    
    # 创建GAT模型
    model = GATModel(
        input_dim=93,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        num_classes=2,
        heads=4
    ).to(device)
    
    print(f"原始图: 节点数={x.size(0)}, 边数={edge_index.size(1)}")
    
    # 测试不同的丢弃率
    for drop_rate in [0.1, 0.3, 0.5]:
        print(f"\n测试丢弃率: {drop_rate}")
        
        try:
            # 扰动图
            perturbed_graph = evaluator.perturb_graph(graph, drop_rate=drop_rate)
            print(f"扰动后: 节点数={perturbed_graph.x.size(0)}, 边数={perturbed_graph.edge_index.size(1)}")
            
            # 测试模型前向传播
            model.eval()
            with torch.no_grad():
                outputs = model(perturbed_graph)
                print(f"模型输出形状: {outputs.shape}")
                print(f"✅ 丢弃率 {drop_rate} 测试通过")
                
        except Exception as e:
            print(f"❌ 丢弃率 {drop_rate} 测试失败: {str(e)}")
            return False
    
    print("\n🎉 所有mask修复测试通过！")
    return True

if __name__ == "__main__":
    test_mask_fix()
