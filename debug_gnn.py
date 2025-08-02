#!/usr/bin/env python3
"""
诊断GNN模型训练问题
"""
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch_geometric.data import Data, DataLoader as GraphDataLoader
import logging

logging.basicConfig(level=logging.INFO)

def create_debug_subgraphs(features, labels, num_graphs=10, nodes_per_graph=20):
    """创建调试用的子图并检查数据分布"""
    graphs = []
    num_samples = len(features)
    
    print(f"总样本数: {num_samples}")
    print(f"标签分布: {np.bincount(labels)}")
    print(f"标签唯一值: {np.unique(labels)}")
    
    for i in range(num_graphs):
        # 随机采样节点
        if num_samples >= nodes_per_graph:
            indices = np.random.choice(num_samples, nodes_per_graph, replace=False)
        else:
            indices = np.random.choice(num_samples, nodes_per_graph, replace=True)
        
        # 节点特征和标签
        x = torch.FloatTensor(features[indices])
        node_labels = labels[indices]
        
        print(f"图{i}: 节点标签分布 {np.bincount(node_labels)}")
        
        # 创建简单的环形连接
        edge_indices = []
        for j in range(nodes_per_graph):
            next_node = (j + 1) % nodes_per_graph
            edge_indices.extend([[j, next_node], [next_node, j]])
        
        edge_index = torch.LongTensor(edge_indices).t()
        
        # 图级别标签：基于节点标签的多数投票
        graph_label = int(node_labels.mean() > 0.5)
        print(f"图{i}: 图级别标签 = {graph_label} (节点均值: {node_labels.mean():.3f})")
        
        graph = Data(x=x, edge_index=edge_index, y=torch.LongTensor([graph_label]))
        graphs.append(graph)
    
    return graphs

def test_gnn_training():
    """测试GNN训练过程"""
    from trainer import Trainer
    from data_loader import load_raw_data
    from preprocessor import preprocess  
    from features import extract_features
    import yaml
    
    # 加载配置
    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 加载数据
    raw_data = load_raw_data(config['data'])
    train_df, val_df, test_df = preprocess(raw_data, config['data'])
    
    # 特征工程
    train_features = extract_features(train_df, config['features'])
    val_features = extract_features(val_df, config['features'])
    
    # 检查数据
    print("=== 数据检查 ===")
    print(f"训练特征形状: {train_features.shape}")
    print(f"验证特征形状: {val_features.shape}")
    
    y_train = train_df['target'].values
    y_val = val_df['target'].values
    
    print(f"训练标签分布: {np.bincount(y_train)}")
    print(f"验证标签分布: {np.bincount(y_val)}")
    
    # 创建子图
    print("\n=== 创建训练子图 ===")
    train_graphs = create_debug_subgraphs(train_features, y_train, num_graphs=10, nodes_per_graph=20)
    
    print("\n=== 创建验证子图 ===")
    val_graphs = create_debug_subgraphs(val_features, y_val, num_graphs=5, nodes_per_graph=20)
    
    # 检查图标签分布
    train_graph_labels = [g.y.item() for g in train_graphs]
    val_graph_labels = [g.y.item() for g in val_graphs]
    
    print(f"\n训练图标签分布: {np.bincount(train_graph_labels)}")
    print(f"验证图标签分布: {np.bincount(val_graph_labels)}")
    
    # 创建数据加载器
    train_loader = GraphDataLoader(train_graphs, batch_size=4, shuffle=True)
    val_loader = GraphDataLoader(val_graphs, batch_size=4, shuffle=False)
    
    # 测试一个batch
    print("\n=== 测试数据加载 ===")
    for batch in train_loader:
        print(f"Batch节点特征形状: {batch.x.shape}")
        print(f"Batch边索引形状: {batch.edge_index.shape}")
        print(f"Batch标签: {batch.y}")
        print(f"Batch大小: {batch.num_graphs}")
        break
    
    # 创建简单的GCN模型进行测试
    print("\n=== 测试模型前向传播 ===")
    from src.models.gnn_models import GCNModel
    
    model = GCNModel(input_dim=93, hidden_dim=64, num_layers=2, dropout=0.1)
    model.eval()
    
    with torch.no_grad():
        for batch in train_loader:
            outputs = model(batch)
            print(f"模型输出形状: {outputs.shape}")
            print(f"模型输出: {outputs}")
            print(f"输出概率: {torch.softmax(outputs, dim=1)}")
            break

if __name__ == "__main__":
    test_gnn_training()
