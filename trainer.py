"""
训练脚本 - 支持RNN、GNN、Transformer模型的训练和超参数搜索
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.loader import DataLoader as GraphDataLoader
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import yaml
import json
import time
from tqdm import tqdm
import optuna
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# 导入模型
from src.models.rnn_models import RNNModel, LSTMModel, GRUModel, EnhancedRNNModel
from src.models.gnn_models import GCNModel, GATModel, GraphSAGEModel, HybridGNNModel
from src.models.transformer import TemporalTransformer, MultiScaleTransformer

# 导入数据处理模块
from data_loader import load_raw_data
from preprocessor import preprocess, create_sequences, standardize_features
from features import extract_features
from graph_builder import build_graphs, create_graph_dataloader

logger = logging.getLogger(__name__)


class ModelTrainer:
    """模型训练器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_models = {}
        self.training_history = {}

        # 创建检查点目录
        os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

        logger.info(f"使用设备: {self.device}")

    def prepare_data(self) -> Tuple[Any, Any, Any]:
        """准备训练数据"""
        logger.info("开始准备训练数据...")

        # 加载原始数据
        raw_data = load_raw_data(self.config['data'])

        # 预处理
        train_df, val_df, test_df = preprocess(raw_data, self.config['data'])

        # 特征工程
        train_features = extract_features(train_df, self.config['features'])
        val_features = extract_features(val_df, self.config['features'])
        test_features = extract_features(test_df, self.config['features'])

        # 标准化
        train_df_std, val_df_std, test_df_std, scalers = standardize_features(train_df, val_df, test_df)

        logger.info(f"数据准备完成 - 训练集: {train_features.shape}, 验证集: {val_features.shape}, 测试集: {test_features.shape}")

        return (train_df_std, val_df_std, test_df_std), (train_features, val_features, test_features), scalers

    def create_dataloaders(self, data_dfs: Tuple, features: Tuple, model_type: str) -> Tuple:
        """创建数据加载器"""
        train_df, val_df, test_df = data_dfs
        train_features, val_features, test_features = features

        batch_size = self.config['training']['batch_size']

        if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
            # 序列数据 - 使用特征工程后的特征矩阵
            sequence_length = self.config.get('sequence_length', 24)

            # 从特征矩阵创建序列
            from preprocessor import create_sequences_from_features
            
            # 获取目标变量
            y_train = train_df['target'].values
            y_val = val_df['target'].values
            y_test = test_df['target'].values

            # 创建序列
            X_train, y_train = create_sequences_from_features(train_features, y_train, sequence_length)
            X_val, y_val = create_sequences_from_features(val_features, y_val, sequence_length)
            X_test, y_test = create_sequences_from_features(test_features, y_test, sequence_length)

            # 转换为张量
            X_train = torch.FloatTensor(X_train)
            y_train = torch.LongTensor(y_train)
            X_val = torch.FloatTensor(X_val)
            y_val = torch.LongTensor(y_val)
            X_test = torch.FloatTensor(X_test)
            y_test = torch.LongTensor(y_test)

            # 创建数据集
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            test_dataset = TensorDataset(X_test, y_test)

            # 创建数据加载器
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        elif model_type in ['gcn', 'gat', 'graphsage', 'hybrid_gnn']:
            # 图数据 - 创建多个子图进行图级别分类
            batch_size_graph = min(batch_size, 16)  # 图数据使用适中的batch size
            
            from torch_geometric.data import Data, DataLoader as GraphDataLoader
            
            def create_subgraphs(features, labels, num_graphs=100, nodes_per_graph=20):
                """创建多个子图用于训练"""
                graphs = []
                num_samples = len(features)
                
                for i in range(num_graphs):
                    # 随机采样节点
                    if num_samples >= nodes_per_graph:
                        indices = np.random.choice(num_samples, nodes_per_graph, replace=False)
                    else:
                        indices = np.random.choice(num_samples, nodes_per_graph, replace=True)
                    
                    # 节点特征和标签
                    x = torch.FloatTensor(features[indices])
                    node_labels = labels[indices]
                    
                    # 创建k-近邻图结构
                    edge_indices = []
                    k = min(6, nodes_per_graph - 1)  # 每个节点连接6个邻居
                    
                    for j in range(nodes_per_graph):
                        # 计算与其他节点的距离（欧氏距离）
                        distances = []
                        for k_idx in range(nodes_per_graph):
                            if j != k_idx:
                                dist = np.linalg.norm(features[indices[j]] - features[indices[k_idx]])
                                distances.append((k_idx, dist))
                        
                        # 连接最近的k个邻居
                        distances.sort(key=lambda x: x[1])
                        for neighbor_idx, _ in distances[:k]:
                            edge_indices.extend([[j, neighbor_idx], [neighbor_idx, j]])
                    
                    # 去重并创建边索引
                    edge_indices = list(set(tuple(edge) for edge in edge_indices))
                    if edge_indices:
                        edge_index = torch.LongTensor(edge_indices).t()
                    else:
                        # 如果没有边，创建一些随机连接
                        edge_indices = []
                        for j in range(min(nodes_per_graph, 5)):
                            for k_idx in range(j+1, min(nodes_per_graph, j+3)):
                                edge_indices.extend([[j, k_idx], [k_idx, j]])
                        edge_index = torch.LongTensor(edge_indices).t() if edge_indices else torch.LongTensor([[0], [0]])
                    
                    # 图级别标签：使用更合理的策略
                    # 方案1：如果正样本比例超过20%则为高风险图
                    # 方案2：使用分层采样确保标签均衡
                    positive_ratio = node_labels.mean()
                    
                    if i < num_graphs * 0.6:  # 60%的图基于正样本比例
                        graph_label = int(positive_ratio > 0.3)  # 降低阈值
                    else:  # 40%的图强制为正样本，确保标签均衡
                        graph_label = 1 if positive_ratio > 0.1 else 0
                    
                    graph = Data(x=x, edge_index=edge_index, y=torch.LongTensor([graph_label]))
                    graphs.append(graph)
                
                return graphs
            
            # 创建训练、验证、测试子图
            y_train = train_df['target'].values
            y_val = val_df['target'].values  
            y_test = test_df['target'].values
            
            train_graphs = create_subgraphs(train_features, y_train, num_graphs=200, nodes_per_graph=25)
            val_graphs = create_subgraphs(val_features, y_val, num_graphs=50, nodes_per_graph=25)
            test_graphs = create_subgraphs(test_features, y_test, num_graphs=50, nodes_per_graph=25)

            # 创建图数据加载器
            train_loader = GraphDataLoader(train_graphs, batch_size=batch_size_graph, shuffle=True)
            val_loader = GraphDataLoader(val_graphs, batch_size=batch_size_graph, shuffle=False)
            test_loader = GraphDataLoader(test_graphs, batch_size=batch_size_graph, shuffle=False)

        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        return train_loader, val_loader, test_loader

    def create_model(self, model_type: str, input_dim: int) -> nn.Module:
        """创建模型实例"""
        model_config = self.config['models']

        if model_type == 'rnn':
            model = RNNModel(
                input_dim=input_dim,
                hidden_dim=model_config['rnn']['hidden_dim'],
                num_layers=model_config['rnn']['num_layers'],
                dropout=model_config['rnn']['dropout'],
                bidirectional=model_config['rnn']['bidirectional']
            )
        elif model_type == 'lstm':
            model = LSTMModel(
                input_dim=input_dim,
                hidden_dim=model_config['rnn']['hidden_dim'],
                num_layers=model_config['rnn']['num_layers'],
                dropout=model_config['rnn']['dropout'],
                bidirectional=model_config['rnn']['bidirectional']
            )
        elif model_type == 'gru':
            model = GRUModel(
                input_dim=input_dim,
                hidden_dim=model_config['rnn']['hidden_dim'],
                num_layers=model_config['rnn']['num_layers'],
                dropout=model_config['rnn']['dropout'],
                bidirectional=model_config['rnn']['bidirectional']
            )
        elif model_type == 'gcn':
            model = GCNModel(
                input_dim=input_dim,
                hidden_dim=model_config['gnn']['hidden_dim'],
                num_layers=model_config['gnn']['num_layers'],
                dropout=model_config['gnn']['dropout']
            )
        elif model_type == 'gat':
            model = GATModel(
                input_dim=input_dim,
                hidden_dim=model_config['gnn']['hidden_dim'],
                num_layers=model_config['gnn']['num_layers'],
                dropout=model_config['gnn']['dropout'],
                heads=model_config['gnn']['heads']
            )
        elif model_type == 'graphsage':
            model = GraphSAGEModel(
                input_dim=input_dim,
                hidden_dim=model_config['gnn']['hidden_dim'],
                num_layers=model_config['gnn']['num_layers'],
                dropout=model_config['gnn']['dropout']
            )
        elif model_type == 'transformer':
            # 检查 transformer 配置
            if 'transformer' not in model_config:
                logger.error("配置文件中缺少 transformer 配置")
                raise ValueError("配置文件中缺少 transformer 配置")
                
            model = TemporalTransformer(
                input_dim=input_dim,
                d_model=model_config['transformer']['d_model'],
                nhead=model_config['transformer']['nhead'],
                num_layers=model_config['transformer']['num_layers'],
                dropout=model_config['transformer']['dropout']
            )
        elif model_type == 'enhanced_rnn':
            model = EnhancedRNNModel(
                input_dim=input_dim,
                hidden_dim=model_config['rnn']['hidden_dim'],
                num_layers=model_config['rnn']['num_layers'],
                dropout=model_config['rnn']['dropout'],
                rnn_type='LSTM'
            )
        elif model_type == 'hybrid_gnn':
            model = HybridGNNModel(
                input_dim=input_dim,
                hidden_dim=model_config['gnn']['hidden_dim'],
                num_layers=model_config['gnn']['num_layers'],
                dropout=model_config['gnn']['dropout']
            )
        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        return model.to(self.device)

    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   optimizer: optim.Optimizer, criterion: nn.Module) -> Dict[str, float]:
        """训练一个epoch"""
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="训练中")
        for batch_idx, batch in enumerate(pbar):
            optimizer.zero_grad()

            if isinstance(batch, (list, tuple)):
                # 序列数据
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = model(inputs)
            else:
                # 图数据
                batch = batch.to(self.device)
                outputs = model(batch)
                targets = batch.y

            loss = criterion(outputs, targets)
            
            # 检查 nan
            if torch.isnan(loss):
                logger.warning(f"检测到 nan loss，跳过此 batch")
                continue
                
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })

        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        return {'loss': avg_loss, 'accuracy': accuracy}

    def validate_epoch(self, model: nn.Module, val_loader: DataLoader,
                      criterion: nn.Module) -> Dict[str, float]:
        """验证一个epoch"""
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []

        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    # 序列数据
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                else:
                    # 图数据
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    targets = batch.y

                loss = criterion(outputs, targets)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                # 收集预测结果用于计算更多指标
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

                # 获取概率
                probabilities = torch.softmax(outputs, dim=1)
                all_probabilities.extend(probabilities.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        accuracy = 100. * correct / total

        # 计算F1和AUC
        f1 = f1_score(all_targets, all_predictions, average='weighted')

        try:
            if len(np.unique(all_targets)) == 2:  # 二分类
                auc = roc_auc_score(all_targets, np.array(all_probabilities)[:, 1])
            else:  # 多分类
                auc = roc_auc_score(all_targets, all_probabilities, multi_class='ovr')
        except:
            auc = 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'f1': f1,
            'auc': auc
        }

    def train_model(self, model_type: str, train_loader: DataLoader,
                   val_loader: DataLoader, input_dim: int) -> Dict[str, Any]:
        """训练单个模型"""
        logger.info(f"开始训练 {model_type} 模型...")

        # 创建模型
        model = self.create_model(model_type, input_dim)

        # 优化器和损失函数
        optimizer = optim.Adam(
            model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()

        # 学习率调度器
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # 早停
        best_val_loss = float('inf')
        patience_counter = 0
        patience = self.config['training']['early_stopping']

        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_f1': [], 'val_auc': []
        }

        # 训练循环
        for epoch in range(self.config['training']['epochs']):
            start_time = time.time()

            # 训练
            train_metrics = self.train_epoch(model, train_loader, optimizer, criterion)

            # 验证
            val_metrics = self.validate_epoch(model, val_loader, criterion)

            # 学习率调度
            scheduler.step(val_metrics['loss'])

            # 记录历史
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_auc'].append(val_metrics['auc'])

            epoch_time = time.time() - start_time

            logger.info(
                f"Epoch {epoch+1}/{self.config['training']['epochs']} "
                f"({epoch_time:.2f}s) - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Train Acc: {train_metrics['accuracy']:.2f}%, "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.2f}%, "
                f"Val F1: {val_metrics['f1']:.4f}, "
                f"Val AUC: {val_metrics['auc']:.4f}"
            )

            # 早停检查
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0

                # 保存最佳模型
                checkpoint_path = os.path.join(
                    self.config['training']['checkpoint_dir'],
                    f'best_{model_type}_model.pth'
                )
                model.save(checkpoint_path)
                self.best_models[model_type] = checkpoint_path

            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                    break

        self.training_history[model_type] = history

        return {
            'model_type': model_type,
            'best_checkpoint': self.best_models[model_type],
            'final_val_loss': best_val_loss,
            'history': history
        }

    def train_all_models(self, model_list: Optional[List[str]] = None) -> Dict[str, Any]:
        """训练所有模型或指定模型列表"""
        if model_list is None:
            model_list = ['lstm', 'gru', 'gcn', 'gat', 'graphsage', 'transformer']

        # 准备数据
        data_dfs, features, scalers = self.prepare_data()

        results = {}

        for model_type in model_list:
            try:
                # 创建数据加载器
                train_loader, val_loader, test_loader = self.create_dataloaders(
                    data_dfs, features, model_type
                )

                # 确定输入维度
                if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
                    input_dim = features[0].shape[1]  # 序列特征维度
                else:  # GNN模型
                    input_dim = self.config['models']['gnn']['input_dim']

                # 训练模型
                result = self.train_model(model_type, train_loader, val_loader, input_dim)
                results[model_type] = result

                logger.info(f"✅ {model_type} 模型训练完成")

            except Exception as e:
                logger.error(f"❌ {model_type} 模型训练失败: {str(e)}")
                results[model_type] = {'error': str(e)}

        # 保存训练结果
        self.save_training_results(results)

        return results

    def save_training_results(self, results: Dict[str, Any]):
        """保存训练结果"""
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)

        # 保存详细结果
        with open(os.path.join(results_dir, 'training_results.json'), 'w') as f:
            # 过滤掉不能序列化的内容
            serializable_results = {}
            for model_type, result in results.items():
                if 'error' not in result:
                    serializable_results[model_type] = {
                        'model_type': result['model_type'],
                        'best_checkpoint': result['best_checkpoint'],
                        'final_val_loss': result['final_val_loss']
                    }
                else:
                    serializable_results[model_type] = result

            json.dump(serializable_results, f, indent=2)

        # 保存训练历史
        with open(os.path.join(results_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)

        logger.info("训练结果已保存到 results/ 目录")


def hyperparameter_optimization(config: Dict, model_type: str, n_trials: int = 50) -> Dict[str, Any]:
    """使用Optuna进行超参数优化"""

    def objective(trial):
        # 建议超参数
        if model_type in ['rnn', 'lstm', 'gru']:
            trial_config = config.copy()
            trial_config['models']['rnn']['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 256, step=32)
            trial_config['models']['rnn']['num_layers'] = trial.suggest_int('num_layers', 1, 4)
            trial_config['models']['rnn']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

        elif model_type in ['gcn', 'gat', 'graphsage']:
            trial_config = config.copy()
            trial_config['models']['gnn']['hidden_dim'] = trial.suggest_int('hidden_dim', 64, 256, step=32)
            trial_config['models']['gnn']['num_layers'] = trial.suggest_int('num_layers', 2, 5)
            trial_config['models']['gnn']['dropout'] = trial.suggest_float('dropout', 0.1, 0.5)
            trial_config['training']['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

            if model_type == 'gat':
                trial_config['models']['gnn']['heads'] = trial.suggest_categorical('heads', [2, 4, 8])

        # 创建训练器并训练
        trainer = ModelTrainer(trial_config)
        results = trainer.train_all_models([model_type])

        # 返回验证损失作为优化目标
        if model_type in results and 'final_val_loss' in results[model_type]:
            return results[model_type]['final_val_loss']
        else:
            return float('inf')

    # 创建研究
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    logger.info("超参数优化完成:")
    logger.info(f"最佳参数: {study.best_params}")
    logger.info(f"最佳值: {study.best_value}")

    return {
        'best_params': study.best_params,
        'best_value': study.best_value,
        'study': study
    }


def train(config: Dict, model_type: Optional[str] = None):
    """训练入口函数"""
    trainer = ModelTrainer(config)

    if model_type:
        # 训练单个模型
        results = trainer.train_all_models([model_type])
    else:
        # 训练所有模型
        results = trainer.train_all_models()

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='供应链风险预测模型训练')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--model', type=str, help='指定训练的模型类型')
    parser.add_argument('--optimize', action='store_true', help='启用超参数优化')
    parser.add_argument('--trials', type=int, default=50, help='超参数优化试验次数')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('logs/training.log'),
            logging.StreamHandler()
        ]
    )

    os.makedirs('logs', exist_ok=True)

    if args.optimize:
        # 超参数优化
        if not args.model:
            print("超参数优化需要指定模型类型 (--model)")
            exit(1)

        logger.info(f"开始 {args.model} 模型的超参数优化...")
        optimization_results = hyperparameter_optimization(config, args.model, args.trials)

        # 保存优化结果
        with open(f'results/hyperopt_{args.model}.json', 'w') as f:
            json.dump({
                'best_params': optimization_results['best_params'],
                'best_value': optimization_results['best_value']
            }, f, indent=2)

    else:
        # 正常训练
        logger.info("开始模型训练...")
        results = train(config, args.model)

        logger.info("训练完成!")
        for model_type, result in results.items():
            if 'error' not in result:
                logger.info(f"{model_type}: 最佳验证损失 = {result['final_val_loss']:.4f}")
            else:
                logger.error(f"{model_type}: 训练失败 - {result['error']}")
