#!/usr/bin/env python3
"""
Debug GNN评估错误 - 修复版本
"""

import traceback
import sys
import os
import logging
import torch

# 设置详细日志
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def fix_dropout_config(model_config):
    """修复checkpoint中的dropout参数"""
    if 'dropout' in model_config:
        dropout_val = model_config['dropout']
        if hasattr(dropout_val, 'p'):  # 如果是Dropout对象
            model_config['dropout'] = dropout_val.p
            print(f"   修复dropout参数: {dropout_val} -> {dropout_val.p}")
    return model_config

def test_gnn_model_loading():
    """测试GNN模型加载"""
    try:
        import yaml
        from src.models.gnn_models import GATModel, GCNModel, GraphSAGEModel
        from trainer import ModelTrainer
        from graph_builder import build_graphs
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("开始测试GNN模型加载...")
        
        # 准备训练器（获取数据）
        print("1. 准备训练数据...")
        trainer = ModelTrainer(config)
        trainer.prepare_data()
        print("   数据准备完成")
        
        # 测试GAT模型
        print("\n2. 测试GAT模型...")
        gat_checkpoint_path = 'checkpoints/best_gat_model.pth'
        if os.path.exists(gat_checkpoint_path):
            try:
                checkpoint = torch.load(gat_checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_config' in checkpoint:
                    model_config = fix_dropout_config(checkpoint['model_config'])
                    gat_model = GATModel(**model_config)
                    gat_model.load_state_dict(checkpoint['model_state_dict'])
                    print("   GAT模型加载成功")
                    
                    # 测试前向传播
                    # 创建一个简单的测试图
                    import pandas as pd
                    test_df = pd.DataFrame({
                        'feature_' + str(i): trainer.X_test.iloc[:100, i] 
                        for i in range(min(10, trainer.X_test.shape[1]))
                    })
                    test_df['risk_flag'] = trainer.y_test.iloc[:100]
                    
                    # 使用build_graphs函数构建图
                    test_graphs = build_graphs(test_df, config)
                    if test_graphs:
                        test_graph = test_graphs[0].to_torch_geometric()
                    
                    gat_model.eval()
                    with torch.no_grad():
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        gat_model = gat_model.to(device)
                        test_graph = test_graph.to(device)
                        
                        outputs = gat_model(test_graph)
                        print(f"   GAT前向传播成功: 输出形状 {outputs.shape}")
                else:
                    print("   GAT checkpoint中没有模型配置")
            except Exception as e:
                print(f"   GAT模型测试失败: {e}")
                traceback.print_exc()
        else:
            print("   GAT checkpoint不存在")
        
        # 测试GCN模型
        print("\n3. 测试GCN模型...")
        gcn_checkpoint_path = 'checkpoints/best_gcn_model.pth'
        if os.path.exists(gcn_checkpoint_path):
            try:
                checkpoint = torch.load(gcn_checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_config' in checkpoint:
                    model_config = fix_dropout_config(checkpoint['model_config'])
                    gcn_model = GCNModel(**model_config)
                    gcn_model.load_state_dict(checkpoint['model_state_dict'])
                    print("   GCN模型加载成功")
                else:
                    print("   GCN checkpoint中没有模型配置")
            except Exception as e:
                print(f"   GCN模型测试失败: {e}")
                traceback.print_exc()
        else:
            print("   GCN checkpoint不存在")
        
        # 测试GraphSAGE模型
        print("\n4. 测试GraphSAGE模型...")
        sage_checkpoint_path = 'checkpoints/best_graphsage_model.pth'
        if os.path.exists(sage_checkpoint_path):
            try:
                checkpoint = torch.load(sage_checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_config' in checkpoint:
                    model_config = fix_dropout_config(checkpoint['model_config'])
                    sage_model = GraphSAGEModel(**model_config)
                    sage_model.load_state_dict(checkpoint['model_state_dict'])
                    print("   GraphSAGE模型加载成功")
                else:
                    print("   GraphSAGE checkpoint中没有模型配置")
            except Exception as e:
                print(f"   GraphSAGE模型测试失败: {e}")
                traceback.print_exc()
        else:
            print("   GraphSAGE checkpoint不存在")
        
        print("\n5. GNN模型加载测试完成！")
        
    except Exception as e:
        print(f"整体测试失败: {e}")
        traceback.print_exc()

def test_evaluator():
    """测试评估器"""
    try:
        print("\n" + "="*50)
        print("测试评估器...")
        
        from evaluator import evaluate
        import yaml
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 只测试GAT模型的评估
        if os.path.exists('checkpoints/best_gat_model.pth'):
            print("测试GAT模型评估...")
            try:
                model_paths = {'gat': 'checkpoints/best_gat_model.pth'}
                results = evaluate(model_paths, config)
                print("GAT评估成功:", results.get('gat', {}).keys() if results.get('gat') else 'No results')
            except Exception as e:
                print(f"GAT评估失败: {e}")
                traceback.print_exc()
        else:
            print("GAT checkpoint不存在，跳过评估测试")
            
    except Exception as e:
        print(f"评估器测试失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_gnn_model_loading()
    test_evaluator()
