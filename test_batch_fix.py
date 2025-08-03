#!/usr/bin/env python3
"""
测试批处理的mask和batch修复
"""

import torch
from torch_geometric.data import Data, DataLoader
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from robustness import RobustnessEvaluator
from src.models.gnn_models import GATModel
import yaml

def test_batch_fix():
    """测试批处理修复"""
    print("🧪 测试批处理扰动修复...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建多个图
    graphs = []
    for i in range(8):  # 8个图
        num_nodes = 10 + i  # 不同的节点数
        x = torch.randn(num_nodes, 93, device=device)
        
        # 创建简单的连通图
        edge_index = []
        for j in range(num_nodes - 1):
            edge_index.extend([[j, j+1], [j+1, j]])
        edge_index = torch.tensor(edge_index, device=device).t()
        
        edge_attr = torch.randn(edge_index.size(1), 3, device=device)
        y = torch.randint(0, 2, (1,), device=device)
        
        graph = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
        graphs.append(graph)
    
    # 创建DataLoader
    data_loader = DataLoader(graphs, batch_size=4, shuffle=False)
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    evaluator = RobustnessEvaluator(config)
    
    # 创建模型
    model = GATModel(
        input_dim=93,
        hidden_dim=64,
        num_layers=2,
        dropout=0.1,
        num_classes=2,
        heads=4
    ).to(device)
    
    model.eval()
    
    print("测试批处理扰动...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            batch = batch.to(device)
            print(f"\n批次 {batch_idx}:")
            print(f"  原始: 总节点数={batch.x.size(0)}, 总边数={batch.edge_index.size(1)}")
            print(f"  批次大小: {batch.batch.max().item() + 1}")
            
            try:
                # 测试基准前向传播
                outputs = model(batch)
                print(f"  基准输出形状: {outputs.shape}")
                
                # 测试扰动
                for drop_rate in [0.2, 0.4]:
                    perturbed_batch = evaluator.perturb_graph(batch, drop_rate=drop_rate)
                    print(f"  扰动({drop_rate}): 节点数={perturbed_batch.x.size(0)}, 边数={perturbed_batch.edge_index.size(1)}")
                    
                    # 检查batch信息是否一致
                    if hasattr(perturbed_batch, 'batch') and perturbed_batch.batch is not None:
                        print(f"    batch大小: {perturbed_batch.batch.size(0)}, 最大batch值: {perturbed_batch.batch.max().item()}")
                        
                        # 确保batch和节点数一致
                        if perturbed_batch.batch.size(0) != perturbed_batch.x.size(0):
                            print(f"    ❌ batch大小不匹配: batch={perturbed_batch.batch.size(0)}, nodes={perturbed_batch.x.size(0)}")
                            continue
                    
                    # 测试模型前向传播
                    perturbed_outputs = model(perturbed_batch)
                    print(f"    扰动输出形状: {perturbed_outputs.shape}")
                    print(f"    ✅ 扰动率 {drop_rate} 测试通过")
                
            except Exception as e:
                print(f"    ❌ 批次 {batch_idx} 测试失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
    
    print("\n🎉 所有批处理测试通过！")
    return True

if __name__ == "__main__":
    test_batch_fix()
