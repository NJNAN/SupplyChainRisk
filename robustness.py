"""
鲁棒性测试模块 - 测试模型对噪声和扰动的抗干扰能力
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score
import random
import copy
import warnings
warnings.filterwarnings('ignore')

from torch_geometric.data import Data, Batch
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class RobustnessEvaluator:
    """鲁棒性评估器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # 扰动参数
        self.drop_rates = config['robustness']['drop_rates']
        self.shuffle_rates = config['robustness']['shuffle_rates']

    def perturb_sequence(self, data: torch.Tensor, drop_rate: float = 0.0,
                        shuffle_rate: float = 0.0) -> torch.Tensor:
        """
        对序列数据进行扰动

        Args:
            data: 输入序列 [batch_size, seq_len, features]
            drop_rate: 节点丢弃率
            shuffle_rate: 顺序打乱率

        Returns:
            扰动后的序列
        """
        if drop_rate == 0.0 and shuffle_rate == 0.0:
            return data

        perturbed_data = data.clone()
        batch_size, seq_len, features = data.shape

        for batch_idx in range(batch_size):
            sequence = perturbed_data[batch_idx]  # [seq_len, features]

            # 节点丢弃
            if drop_rate > 0:
                drop_mask = torch.rand(seq_len) > drop_rate
                if drop_mask.sum() > 0:  # 确保至少保留一个节点
                    # 将丢弃的节点设为零或用相邻节点填充
                    dropped_indices = torch.where(~drop_mask)[0]
                    for idx in dropped_indices:
                        if idx > 0:
                            sequence[idx] = sequence[idx - 1]  # 用前一个节点填充
                        elif idx < seq_len - 1:
                            sequence[idx] = sequence[idx + 1]  # 用后一个节点填充
                        else:
                            sequence[idx] = 0  # 如果只有一个节点，设为零

            # 顺序打乱
            if shuffle_rate > 0:
                num_shuffles = int(seq_len * shuffle_rate)
                if num_shuffles > 0:
                    # 随机选择要打乱的位置
                    shuffle_indices = random.sample(range(seq_len), min(num_shuffles, seq_len))
                    if len(shuffle_indices) > 1:
                        # 打乱选中的位置
                        shuffled_values = sequence[shuffle_indices].clone()
                        random.shuffle(shuffle_indices)
                        sequence[shuffle_indices] = shuffled_values

            perturbed_data[batch_idx] = sequence

        return perturbed_data

    def perturb_graph(self, graph_data: Union[Data, List[Data]], drop_rate: float = 0.0) -> Union[Data, List[Data]]:
        """
        对图数据进行扰动

        Args:
            graph_data: 图数据或图数据列表
            drop_rate: 节点/边丢弃率

        Returns:
            扰动后的图数据
        """
        if drop_rate == 0.0:
            return graph_data

        if isinstance(graph_data, list):
            return [self._perturb_single_graph(g, drop_rate) for g in graph_data]
        else:
            return self._perturb_single_graph(graph_data, drop_rate)

    def _perturb_single_graph(self, graph: Data, drop_rate: float) -> Data:
        """对单个图进行扰动"""
        if drop_rate == 0.0:
            return graph

        perturbed_graph = graph.clone()
        num_nodes = graph.x.size(0)
        num_edges = graph.edge_index.size(1)

        # 节点丢弃
        if drop_rate > 0 and num_nodes > 1:
            keep_nodes = int(num_nodes * (1 - drop_rate))
            keep_nodes = max(1, keep_nodes)  # 至少保留一个节点

            # 随机选择要保留的节点
            keep_indices = torch.randperm(num_nodes)[:keep_nodes]
            keep_indices = torch.sort(keep_indices)[0]

            # 更新节点特征
            perturbed_graph.x = graph.x[keep_indices]

            # 更新边索引
            if hasattr(graph, 'edge_index') and graph.edge_index.size(1) > 0:
                # 创建节点映射
                node_mapping = {old_idx.item(): new_idx for new_idx, old_idx in enumerate(keep_indices)}

                # 过滤边
                valid_edges = []
                for i in range(num_edges):
                    src, dst = graph.edge_index[:, i].tolist()
                    if src in node_mapping and dst in node_mapping:
                        new_src = node_mapping[src]
                        new_dst = node_mapping[dst]
                        valid_edges.append([new_src, new_dst])

                if valid_edges:
                    perturbed_graph.edge_index = torch.tensor(valid_edges, dtype=torch.long).t()

                    # 更新边属性
                    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                        valid_edge_indices = []
                        for i, (src, dst) in enumerate(graph.edge_index.t().tolist()):
                            if src in node_mapping and dst in node_mapping:
                                valid_edge_indices.append(i)

                        if valid_edge_indices:
                            perturbed_graph.edge_attr = graph.edge_attr[valid_edge_indices]
                        else:
                            perturbed_graph.edge_attr = None
                else:
                    # 如果没有有效边，创建自环
                    perturbed_graph.edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
                        perturbed_graph.edge_attr = graph.edge_attr[:1]  # 保留一个边属性

        return perturbed_graph

    def add_noise_to_sequence(self, data: torch.Tensor, noise_level: float = 0.1,
                            noise_type: str = 'gaussian') -> torch.Tensor:
        """
        为序列数据添加噪声

        Args:
            data: 输入序列
            noise_level: 噪声强度
            noise_type: 噪声类型 ('gaussian', 'uniform', 'dropout')
        """
        if noise_level == 0.0:
            return data

        noisy_data = data.clone()

        if noise_type == 'gaussian':
            noise = torch.randn_like(data) * noise_level
            noisy_data = data + noise

        elif noise_type == 'uniform':
            noise = (torch.rand_like(data) - 0.5) * 2 * noise_level
            noisy_data = data + noise

        elif noise_type == 'dropout':
            dropout_mask = torch.rand_like(data) > noise_level
            noisy_data = data * dropout_mask.float()

        return noisy_data

    def simulate_business_disruptions(self, data: torch.Tensor, disruption_type: str = 'holiday') -> torch.Tensor:
        """
        模拟业务中断场景

        Args:
            data: 输入数据
            disruption_type: 中断类型 ('holiday', 'port_closure', 'weather', 'pandemic')
        """
        disrupted_data = data.clone()
        batch_size, seq_len, features = data.shape

        if disruption_type == 'holiday':
            # 模拟节假日停运：随机将某些时间段的特征设为零
            for batch_idx in range(batch_size):
                # 随机选择1-3个连续的时间段
                num_disruptions = random.randint(1, 3)
                for _ in range(num_disruptions):
                    start_idx = random.randint(0, seq_len - 5)
                    end_idx = min(start_idx + random.randint(2, 7), seq_len)  # 2-7天的中断

                    # 将这个时间段的某些特征降低
                    disrupted_data[batch_idx, start_idx:end_idx, :features//2] *= 0.1

        elif disruption_type == 'port_closure':
            # 模拟港口关闭：影响物流相关特征
            for batch_idx in range(batch_size):
                closure_start = random.randint(0, seq_len - 10)
                closure_end = min(closure_start + random.randint(5, 15), seq_len)

                # 物流特征受到严重影响
                disrupted_data[batch_idx, closure_start:closure_end, features//3:2*features//3] *= 0.05

        elif disruption_type == 'weather':
            # 模拟极端天气：短期但强烈的影响
            for batch_idx in range(batch_size):
                weather_events = random.randint(1, 4)
                for _ in range(weather_events):
                    event_start = random.randint(0, seq_len - 3)
                    event_end = min(event_start + random.randint(1, 5), seq_len)

                    # 天气影响运输效率
                    weather_impact = random.uniform(0.3, 0.8)
                    disrupted_data[batch_idx, event_start:event_end] *= weather_impact

        elif disruption_type == 'pandemic':
            # 模拟疫情影响：长期且全面的影响
            pandemic_start = random.randint(0, seq_len//3)
            pandemic_end = min(pandemic_start + random.randint(seq_len//2, seq_len), seq_len)

            impact_factor = random.uniform(0.4, 0.7)
            disrupted_data[:, pandemic_start:pandemic_end] *= impact_factor

        return disrupted_data

    def evaluate_model_robustness(self, model: nn.Module, test_loader, model_type: str,
                                perturbation_type: str = 'drop') -> Dict[str, Any]:
        """
        评估模型鲁棒性

        Args:
            model: 待测试的模型
            test_loader: 测试数据加载器
            model_type: 模型类型
            perturbation_type: 扰动类型
        """
        model.eval()

        # 基准性能（无扰动）
        baseline_acc, baseline_f1 = self._evaluate_performance(model, test_loader, model_type)

        results = {
            'model_type': model_type,
            'perturbation_type': perturbation_type,
            'baseline_accuracy': baseline_acc,
            'baseline_f1': baseline_f1,
            'robustness_curves': {}
        }

        if perturbation_type == 'drop':
            # 测试不同丢弃率
            perturbation_levels = self.drop_rates
            for level in perturbation_levels:
                acc, f1 = self._evaluate_with_perturbation(
                    model, test_loader, model_type, 'drop', level
                )
                results['robustness_curves'][f'drop_{level}'] = {
                    'accuracy': acc,
                    'f1': f1,
                    'accuracy_drop': baseline_acc - acc,
                    'f1_drop': baseline_f1 - f1
                }

        elif perturbation_type == 'shuffle':
            # 测试不同打乱率
            perturbation_levels = self.shuffle_rates
            for level in perturbation_levels:
                acc, f1 = self._evaluate_with_perturbation(
                    model, test_loader, model_type, 'shuffle', level
                )
                results['robustness_curves'][f'shuffle_{level}'] = {
                    'accuracy': acc,
                    'f1': f1,
                    'accuracy_drop': baseline_acc - acc,
                    'f1_drop': baseline_f1 - f1
                }

        elif perturbation_type == 'noise':
            # 测试不同噪声水平
            noise_levels = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
            for level in noise_levels:
                acc, f1 = self._evaluate_with_perturbation(
                    model, test_loader, model_type, 'noise', level
                )
                results['robustness_curves'][f'noise_{level}'] = {
                    'accuracy': acc,
                    'f1': f1,
                    'accuracy_drop': baseline_acc - acc,
                    'f1_drop': baseline_f1 - f1
                }

        elif perturbation_type == 'business':
            # 测试业务中断场景
            disruption_types = ['holiday', 'port_closure', 'weather', 'pandemic']
            for disruption in disruption_types:
                acc, f1 = self._evaluate_with_business_disruption(
                    model, test_loader, model_type, disruption
                )
                results['robustness_curves'][f'disruption_{disruption}'] = {
                    'accuracy': acc,
                    'f1': f1,
                    'accuracy_drop': baseline_acc - acc,
                    'f1_drop': baseline_f1 - f1
                }

        return results

    def _evaluate_performance(self, model: nn.Module, test_loader, model_type: str) -> Tuple[float, float]:
        """评估模型性能"""
        all_predictions = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
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

                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        return accuracy, f1

    def _evaluate_with_perturbation(self, model: nn.Module, test_loader, model_type: str,
                                  perturbation_type: str, level: float) -> Tuple[float, float]:
        """使用扰动数据评估模型"""
        all_predictions = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    # 序列数据
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # 应用扰动
                    if perturbation_type == 'drop':
                        perturbed_inputs = self.perturb_sequence(inputs, drop_rate=level)
                    elif perturbation_type == 'shuffle':
                        perturbed_inputs = self.perturb_sequence(inputs, shuffle_rate=level)
                    elif perturbation_type == 'noise':
                        perturbed_inputs = self.add_noise_to_sequence(inputs, noise_level=level)
                    else:
                        perturbed_inputs = inputs

                    outputs = model(perturbed_inputs)

                else:
                    # 图数据
                    batch = batch.to(self.device)

                    # 应用扰动
                    if perturbation_type == 'drop':
                        # 对图数据应用节点丢弃
                        perturbed_batch = self.perturb_graph(batch, drop_rate=level)
                        outputs = model(perturbed_batch)
                    else:
                        outputs = model(batch)

                    targets = batch.y

                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        return accuracy, f1

    def _evaluate_with_business_disruption(self, model: nn.Module, test_loader,
                                         model_type: str, disruption_type: str) -> Tuple[float, float]:
        """使用业务中断场景评估模型"""
        all_predictions = []
        all_targets = []

        model.eval()
        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    # 序列数据
                    inputs, targets = batch
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)

                    # 模拟业务中断
                    disrupted_inputs = self.simulate_business_disruptions(inputs, disruption_type)
                    outputs = model(disrupted_inputs)

                else:
                    # 图数据暂不支持业务中断模拟
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    targets = batch.y

                _, predicted = outputs.max(1)
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())

        accuracy = accuracy_score(all_targets, all_predictions)
        f1 = f1_score(all_targets, all_predictions, average='weighted', zero_division=0)

        return accuracy, f1

    def plot_robustness_curves(self, all_results: Dict[str, Dict]):
        """绘制鲁棒性曲线"""

        perturbation_types = ['drop', 'shuffle', 'noise', 'business']

        for perturbation_type in perturbation_types:
            # 检查是否有该类型的结果
            relevant_results = {}
            for model_name, results in all_results.items():
                if results['perturbation_type'] == perturbation_type:
                    relevant_results[model_name] = results

            if not relevant_results:
                continue

            # 创建图表
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            for model_name, results in relevant_results.items():
                curves = results['robustness_curves']

                # 提取扰动水平和性能
                levels = []
                accuracies = []
                f1_scores = []

                for key, metrics in curves.items():
                    if perturbation_type in key:
                        level = float(key.split('_')[-1])
                        levels.append(level)
                        accuracies.append(metrics['accuracy'])
                        f1_scores.append(metrics['f1'])

                if levels:
                    # 排序
                    sorted_data = sorted(zip(levels, accuracies, f1_scores))
                    levels, accuracies, f1_scores = zip(*sorted_data)

                    # 绘制曲线
                    ax1.plot(levels, accuracies, marker='o', label=f'{model_name}', linewidth=2)
                    ax2.plot(levels, f1_scores, marker='s', label=f'{model_name}', linewidth=2)

            # 设置图表
            if perturbation_type == 'business':
                ax1.set_xlabel('业务中断类型')
                ax2.set_xlabel('业务中断类型')
                # 为业务中断设置分类标签
                disruption_labels = ['holiday', 'port_closure', 'weather', 'pandemic']
                ax1.set_xticks(range(len(disruption_labels)))
                ax1.set_xticklabels(disruption_labels, rotation=45)
                ax2.set_xticks(range(len(disruption_labels)))
                ax2.set_xticklabels(disruption_labels, rotation=45)
            else:
                ax1.set_xlabel(f'扰动强度 ({perturbation_type})')
                ax2.set_xlabel(f'扰动强度 ({perturbation_type})')

            ax1.set_ylabel('准确率')
            ax2.set_ylabel('F1分数')
            ax1.set_title(f'准确率 vs {perturbation_type.upper()} 扰动强度')
            ax2.set_title(f'F1分数 vs {perturbation_type.upper()} 扰动强度')
            ax1.legend()
            ax2.legend()
            ax1.grid(True, alpha=0.3)
            ax2.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, f'robustness_curves_{perturbation_type}.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def generate_robustness_report(self, all_results: Dict[str, Dict]) -> str:
        """生成鲁棒性报告"""
        report = []
        report.append("# 模型鲁棒性评估报告\n")
        report.append(f"评估时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 按扰动类型分组结果
        perturbation_groups = {}
        for model_name, results in all_results.items():
            pert_type = results['perturbation_type']
            if pert_type not in perturbation_groups:
                perturbation_groups[pert_type] = {}
            perturbation_groups[pert_type][model_name] = results

        for pert_type, models in perturbation_groups.items():
            report.append(f"## {pert_type.upper()} 扰动测试结果\n")

            # 基准性能
            report.append("### 基准性能\n")
            report.append("| 模型 | 准确率 | F1分数 |\n")
            report.append("|------|--------|--------|\n")

            for model_name, results in models.items():
                baseline_acc = results['baseline_accuracy']
                baseline_f1 = results['baseline_f1']
                report.append(f"| {model_name} | {baseline_acc:.4f} | {baseline_f1:.4f} |\n")

            report.append("\n")

            # 鲁棒性分析
            report.append("### 鲁棒性分析\n")

            for model_name, results in models.items():
                report.append(f"**{model_name}**:\n")

                curves = results['robustness_curves']
                max_acc_drop = 0
                max_f1_drop = 0
                worst_scenario = ""

                for scenario, metrics in curves.items():
                    acc_drop = metrics['accuracy_drop']
                    f1_drop = metrics['f1_drop']

                    if acc_drop > max_acc_drop:
                        max_acc_drop = acc_drop
                        worst_scenario = scenario

                    max_f1_drop = max(max_f1_drop, f1_drop)

                report.append(f"- 最大准确率下降: {max_acc_drop:.2%} (场景: {worst_scenario})\n")
                report.append(f"- 最大F1下降: {max_f1_drop:.2%}\n")

                # 鲁棒性评级
                if max_acc_drop < 0.05:
                    robustness_level = "优秀"
                elif max_acc_drop < 0.1:
                    robustness_level = "良好"
                elif max_acc_drop < 0.2:
                    robustness_level = "一般"
                else:
                    robustness_level = "较差"

                report.append(f"- 鲁棒性评级: **{robustness_level}**\n\n")

        # 模型排名
        report.append("## 鲁棒性排名\n")

        # 计算综合鲁棒性分数
        model_scores = {}
        for model_name in all_results.keys():
            total_score = 0
            count = 0

            for results in all_results.values():
                if results['model_type'] == model_name.split('_')[0]:  # 匹配模型类型
                    for metrics in results['robustness_curves'].values():
                        total_score += (1 - metrics['accuracy_drop'])  # 性能下降越小越好
                        count += 1

            if count > 0:
                model_scores[model_name] = total_score / count

        # 排序
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

        report.append("| 排名 | 模型 | 鲁棒性分数 |\n")
        report.append("|------|------|------------|\n")

        for rank, (model_name, score) in enumerate(sorted_models, 1):
            report.append(f"| {rank} | {model_name} | {score:.4f} |\n")

        # 结论和建议
        report.append("\n## 结论和建议\n")

        if sorted_models:
            best_model = sorted_models[0][0]
            report.append(f"**最鲁棒的模型**: {best_model}\n\n")

        report.append("**鲁棒性优化建议**:\n")
        report.append("1. 数据增强: 在训练过程中加入扰动样本\n")
        report.append("2. 正则化: 使用dropout、权重衰减等技术\n")
        report.append("3. 集成学习: 组合多个模型提高鲁棒性\n")
        report.append("4. 对抗训练: 使用对抗样本进行训练\n")
        report.append("5. 业务逻辑: 集成业务规则和专家知识\n")

        return "\n".join(report)

    def save_robustness_results(self, all_results: Dict[str, Dict]):
        """保存鲁棒性测试结果"""

        # 保存详细结果
        with open(os.path.join(self.results_dir, 'robustness_results.json'), 'w') as f:
            json.dump(all_results, f, indent=2)

        # 生成并保存报告
        report = self.generate_robustness_report(all_results)
        with open(os.path.join(self.results_dir, 'robustness_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info("鲁棒性测试结果已保存到 results/ 目录")


def run_robustness_tests(model_paths: Dict[str, str], config: Dict) -> Dict[str, Any]:
    """
    运行鲁棒性测试的入口函数
    """
    evaluator = RobustnessEvaluator(config)
    all_results = {}

    # 测试类型
    perturbation_types = ['drop', 'shuffle', 'noise', 'business']

    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue

        try:
            logger.info(f"开始鲁棒性测试: {model_type}")

            # 加载模型和数据
            from trainer import ModelTrainer
            trainer = ModelTrainer(config)

            # 准备数据
            data_dfs, features, scalers = trainer.prepare_data()
            _, _, test_loader = trainer.create_dataloaders(data_dfs, features, model_type)

            # 创建模型
            if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
                input_dim = features[0].shape[1]
            else:
                input_dim = config['models']['gnn']['input_dim']

            model = trainer.create_model(model_type, input_dim)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            model.to(evaluator.device)

            # 对每种扰动类型进行测试
            for pert_type in perturbation_types:
                result_key = f"{model_type}_{pert_type}"

                results = evaluator.evaluate_model_robustness(
                    model, test_loader, model_type, pert_type
                )

                all_results[result_key] = results
                logger.info(f"✅ {model_type} - {pert_type} 测试完成")

        except Exception as e:
            logger.error(f"❌ {model_type} 鲁棒性测试失败: {str(e)}")

    if all_results:
        # 创建可视化
        evaluator.plot_robustness_curves(all_results)

        # 保存结果
        evaluator.save_robustness_results(all_results)

    return all_results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='模型鲁棒性测试')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    parser.add_argument('--models', type=str, help='模型检查点目录')

    args = parser.parse_args()

    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 查找模型文件
    checkpoint_dir = args.models or config['training']['checkpoint_dir']
    model_paths = {}

    if os.path.exists(checkpoint_dir):
        for model_file in os.listdir(checkpoint_dir):
            if model_file.startswith('best_') and model_file.endswith('_model.pth'):
                model_type = model_file.replace('best_', '').replace('_model.pth', '')
                model_paths[model_type] = os.path.join(checkpoint_dir, model_file)

    if model_paths:
        logger.info(f"找到模型: {list(model_paths.keys())}")
        results = run_robustness_tests(model_paths, config)
        logger.info("鲁棒性测试完成!")
    else:
        logger.error("未找到模型文件")
