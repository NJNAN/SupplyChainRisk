#!/usr/bin/env python3
"""
Debug GNN评估错误 - 定位'<' not supported between instances of 'Dropout' and 'int'
"""

import traceback
import sys
import os
import logging

# 设置详细日志
logging.basicConfig(level=logging.DEBUG)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def debug_gnn_eval():
    """Debug GNN评估问题"""
    try:
        from evaluator import evaluate
        import yaml
        
        # 加载配置
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        print("开始Debug GAT模型评估...")
        
        # 尝试直接调用evaluate函数，只评估GAT
        try:
            # 先尝试加载GAT模型看看有什么问题
            from src.models.gnn_models import GATModel
            from trainer import ModelTrainer
            import torch
            
            print("1. 尝试加载GAT模型...")
            trainer = ModelTrainer(config)
            trainer.prepare_data()
            
            # 尝试加载保存的模型，先从checkpoint获取配置
            checkpoint_path = 'checkpoints/best_gat_model.pth'
            if os.path.exists(checkpoint_path):
                print("2. 加载checkpoint...")
                checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # 从checkpoint获取模型配置
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    print("3. 使用checkpoint中的模型配置:", model_config)
                    
                    # 修复dropout参数的问题
                    if 'dropout' in model_config:
                        dropout_val = model_config['dropout']
                        if hasattr(dropout_val, 'p'):  # 如果是Dropout对象
                            model_config['dropout'] = dropout_val.p
                            print(f"   修复dropout参数: {dropout_val} -> {dropout_val.p}")
                    
                    # 使用checkpoint中的配置创建模型
                    model = GATModel(**model_config)
                else:
                    # 如果没有配置，使用配置文件中的参数
                    print("3. checkpoint中没有模型配置，使用配置文件参数")
                    model = GATModel(
                        input_dim=config['models']['gnn']['input_dim'],
                        hidden_dim=config['models']['gnn']['hidden_dim'],
                        num_layers=config['models']['gnn']['num_layers'],
                        dropout=config['models']['gnn']['dropout'],
                        num_classes=2,
                        heads=config['models']['gnn']['heads']
                    )
                
                print("4. 模型创建成功")
                model.load_state_dict(checkpoint['model_state_dict'])
                print("5. checkpoint加载成功")
                
                # 尝试运行模型评估
                print("6. 测试模型前向传播...")
                from graph_builder import GraphBuilder
                
                graph_builder = GraphBuilder(config)
                # 使用测试数据构建图
                test_graph = graph_builder.build_graph(trainer.X_test, trainer.y_test)
                print(f"   测试图构建成功: {test_graph.num_nodes} 节点, {test_graph.num_edges} 边")
                
                # 测试前向传播
                model.eval()
                with torch.no_grad():
                    try:
                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                        model = model.to(device)
                        test_graph = test_graph.to(device)
                        
                        outputs = model(test_graph)
                        print(f"   前向传播成功: 输出形状 {outputs.shape}")
                        
                        # 测试预测
                        probs = torch.softmax(outputs, dim=1)
                        preds = torch.argmax(probs, dim=1)
                        print(f"   预测成功: {preds.shape}, 预测值范围 [{preds.min()}-{preds.max()}]")
                        
                        print("7. GAT模型评估测试通过！")
                        
                    except Exception as e:
                        print(f"   前向传播失败: {e}")
                        traceback.print_exc()
                
            else:
                print("2. checkpoint不存在，使用配置文件参数创建新模型")
                model = GATModel(
                    input_dim=config['models']['gnn']['input_dim'],
                    hidden_dim=config['models']['gnn']['hidden_dim'],
                    num_layers=config['models']['gnn']['num_layers'],
                    dropout=config['models']['gnn']['dropout'],
                    num_classes=2,
                    heads=config['models']['gnn']['heads']
                )
                
        # 测试完GAT后，继续测试其他GNN模型
        print("\n" + "="*50)
        print("测试其他GNN模型...")
        
        # 测试GCN
        print("测试GCN模型...")
        try:
            from src.models.gnn_models import GCNModel
            gcn_checkpoint_path = 'checkpoints/best_gcn_model.pth'
            if os.path.exists(gcn_checkpoint_path):
                print("GCN checkpoint存在，测试加载...")
                checkpoint = torch.load(gcn_checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    # 修复dropout参数
                    if 'dropout' in model_config and hasattr(model_config['dropout'], 'p'):
                        model_config['dropout'] = model_config['dropout'].p
                    gcn_model = GCNModel(**model_config)
                    gcn_model.load_state_dict(checkpoint['model_state_dict'])
                    print("GCN模型加载成功")
                else:
                    print("GCN checkpoint中没有模型配置")
            else:
                print("GCN checkpoint不存在")
        except Exception as e:
            print(f"GCN模型测试失败: {e}")
        
        # 测试GraphSAGE
        print("测试GraphSAGE模型...")
        try:
            from src.models.gnn_models import GraphSAGEModel
            sage_checkpoint_path = 'checkpoints/best_graphsage_model.pth'
            if os.path.exists(sage_checkpoint_path):
                print("GraphSAGE checkpoint存在，测试加载...")
                checkpoint = torch.load(sage_checkpoint_path, map_location='cpu', weights_only=False)
                if 'model_config' in checkpoint:
                    model_config = checkpoint['model_config']
                    # 修复dropout参数
                    if 'dropout' in model_config and hasattr(model_config['dropout'], 'p'):
                        model_config['dropout'] = model_config['dropout'].p
                    sage_model = GraphSAGEModel(**model_config)
                    sage_model.load_state_dict(checkpoint['model_state_dict'])
                    print("GraphSAGE模型加载成功")
                else:
                    print("GraphSAGE checkpoint中没有模型配置")
            else:
                print("GraphSAGE checkpoint不存在")
        except Exception as e:
            print(f"GraphSAGE模型测试失败: {e}")
            
        print("GNN模型checkpoint测试完成！")
                
        except Exception as e:
            print(f"Debug过程失败: {e}")
            print("详细错误:")
            traceback.print_exc()
            
    except Exception as e:
        print(f"初始化失败: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    debug_gnn_eval()
