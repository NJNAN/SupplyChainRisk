"""
模型压缩模块 - 结构剪枝和后训练量化
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import os
import copy
import time
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ModelCompressor:
    """模型压缩器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.compression_results = {}

    def prune_model(self, model: nn.Module, ratio: float, method: str = 'magnitude') -> nn.Module:
        """
        对模型进行结构剪枝

        Args:
            model: 待剪枝的模型
            ratio: 剪枝比例 (0.0-1.0)
            method: 剪枝方法 ('magnitude', 'random', 'structured')

        Returns:
            剪枝后的模型
        """
        logger.info(f"开始剪枝，比例: {ratio:.1%}, 方法: {method}")

        # 深拷贝模型
        pruned_model = copy.deepcopy(model)

        # 获取可剪枝的参数
        parameters_to_prune = []

        for name, module in pruned_model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                parameters_to_prune.append((module, 'weight'))

        if not parameters_to_prune:
            logger.warning("未找到可剪枝的层")
            return pruned_model

        if method == 'magnitude':
            # 全局幅度剪枝
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=ratio,
            )
        elif method == 'random':
            # 随机剪枝
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.RandomUnstructured,
                amount=ratio,
            )
        elif method == 'structured':
            # 结构化剪枝（按通道剪枝）
            for module, param_name in parameters_to_prune:
                if hasattr(module, param_name):
                    if len(getattr(module, param_name).shape) > 1:
                        # 对于多维参数，使用结构化剪枝
                        prune.ln_structured(
                            module, param_name, amount=ratio, n=2, dim=0
                        )
                    else:
                        # 对于一维参数，使用非结构化剪枝
                        prune.l1_unstructured(module, param_name, amount=ratio)

        # 计算剪枝统计信息
        total_params = 0
        pruned_params = 0

        for name, module in pruned_model.named_modules():
            for param_name, param in module.named_parameters():
                if param_name.endswith('_mask'):
                    continue
                total_params += param.numel()
                if hasattr(module, param_name + '_mask'):
                    mask = getattr(module, param_name + '_mask')
                    pruned_params += (mask == 0).sum().item()

        actual_ratio = pruned_params / total_params if total_params > 0 else 0
        logger.info(f"剪枝完成 - 目标比例: {ratio:.1%}, 实际比例: {actual_ratio:.1%}")

        return pruned_model

    def quantize_model(self, model: nn.Module, bits: int = 8, method: str = 'dynamic') -> nn.Module:
        """
        对模型进行量化

        Args:
            model: 待量化的模型
            bits: 量化位数 (4, 8, 16)
            method: 量化方法 ('dynamic', 'static', 'qat')

        Returns:
            量化后的模型
        """
        logger.info(f"开始量化，位数: {bits}, 方法: {method}")

        # 确保模型在CPU上进行量化
        model_cpu = model.cpu()
        model_cpu.eval()

        if method == 'dynamic':
            # 动态量化
            if bits == 8:
                quantized_model = torch.quantization.quantize_dynamic(
                    model_cpu,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )
            elif bits == 16:
                quantized_model = torch.quantization.quantize_dynamic(
                    model_cpu,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.float16
                )
            else:
                logger.warning(f"动态量化不支持 {bits} 位，使用8位")
                quantized_model = torch.quantization.quantize_dynamic(
                    model_cpu,
                    {nn.Linear, nn.LSTM, nn.GRU},
                    dtype=torch.qint8
                )

        elif method == 'static':
            # 静态量化
            quantized_model = self._static_quantization(model_cpu, bits)

        elif method == 'qat':
            # 量化感知训练（需要重新训练）
            quantized_model = self._quantization_aware_training(model_cpu, bits)

        else:
            logger.error(f"未知的量化方法: {method}")
            return model

        return quantized_model

    def _static_quantization(self, model: nn.Module, bits: int) -> nn.Module:
        """静态量化"""
        # 设置量化配置
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 准备量化
        prepared_model = torch.quantization.prepare(model, inplace=False)

        # 这里应该用校准数据集进行校准，但由于没有数据，我们跳过
        logger.warning("静态量化需要校准数据，跳过校准步骤")

        # 转换为量化模型
        quantized_model = torch.quantization.convert(prepared_model, inplace=False)

        return quantized_model

    def _quantization_aware_training(self, model: nn.Module, bits: int) -> nn.Module:
        """量化感知训练"""
        # 设置QAT配置
        model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')

        # 准备QAT
        prepared_model = torch.quantization.prepare_qat(model, inplace=False)

        logger.warning("量化感知训练需要重新训练模型，返回准备好的模型")

        return prepared_model

    def remove_pruning_masks(self, model: nn.Module) -> nn.Module:
        """
        移除剪枝掩码，使剪枝永久化
        """
        for name, module in model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
            if hasattr(module, 'bias_mask'):
                prune.remove(module, 'bias')

        return model

    def calculate_model_size(self, model: nn.Module) -> Dict[str, Any]:
        """
        计算模型大小和参数统计
        """
        total_params = 0
        trainable_params = 0
        non_zero_params = 0

        for param in model.parameters():
            param_count = param.numel()
            total_params += param_count

            if param.requires_grad:
                trainable_params += param_count

            # 计算非零参数（用于评估剪枝效果）
            non_zero_params += (param != 0).sum().item()

        # 计算模型文件大小（字节）
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        sparsity = 1 - (non_zero_params / total_params) if total_params > 0 else 0

        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_zero_params': non_zero_params,
            'sparsity': sparsity,
            'model_size_mb': model_size_mb,
            'param_size_bytes': param_size,
            'buffer_size_bytes': buffer_size
        }

    def measure_inference_time(self, model: nn.Module, sample_input: torch.Tensor,
                             num_runs: int = 100) -> Dict[str, float]:
        """
        测量推理时间
        """
        model.eval()

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)

        # 测量时间
        times = []

        with torch.no_grad():
            for _ in range(num_runs):
                start_time = time.time()
                _ = model(sample_input)
                end_time = time.time()
                times.append(end_time - start_time)

        times = np.array(times) * 1000  # 转换为毫秒

        return {
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times)
        }

    def evaluate_compressed_model(self, original_model: nn.Module, compressed_model: nn.Module,
                                sample_input: torch.Tensor, test_loader=None) -> Dict[str, Any]:
        """
        评估压缩模型的性能
        """
        results = {}

        # 模型大小比较
        original_stats = self.calculate_model_size(original_model)
        compressed_stats = self.calculate_model_size(compressed_model)

        results['size_comparison'] = {
            'original': original_stats,
            'compressed': compressed_stats,
            'compression_ratio': original_stats['model_size_mb'] / compressed_stats['model_size_mb'] if compressed_stats['model_size_mb'] > 0 else float('inf'),
            'parameter_reduction': 1 - (compressed_stats['total_params'] / original_stats['total_params']) if original_stats['total_params'] > 0 else 0
        }

        # 推理时间比较
        original_time = self.measure_inference_time(original_model, sample_input)
        compressed_time = self.measure_inference_time(compressed_model, sample_input)

        results['speed_comparison'] = {
            'original': original_time,
            'compressed': compressed_time,
            'speedup': original_time['mean_ms'] / compressed_time['mean_ms'] if compressed_time['mean_ms'] > 0 else 1.0
        }

        # 如果提供了测试数据，评估准确性
        if test_loader is not None:
            original_acc = self._evaluate_accuracy(original_model, test_loader)
            compressed_acc = self._evaluate_accuracy(compressed_model, test_loader)

            results['accuracy_comparison'] = {
                'original': original_acc,
                'compressed': compressed_acc,
                'accuracy_drop': original_acc - compressed_acc
            }

        return results

    def _evaluate_accuracy(self, model: nn.Module, test_loader) -> float:
        """评估模型准确性"""
        model.eval()
        # 确保模型在正确的设备上
        model = model.to(self.device)
        
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in test_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, targets = batch
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = model(inputs)
                else:
                    batch = batch.to(self.device)
                    outputs = model(batch)
                    targets = batch.y

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return correct / total if total > 0 else 0.0

    def compress_and_evaluate(self, model: nn.Module, model_name: str,
                            sample_input: torch.Tensor, test_loader=None) -> Dict[str, Any]:
        """
        对模型进行全面的压缩和评估
        """
        logger.info(f"开始压缩和评估模型: {model_name}")

        results = {
            'model_name': model_name,
            'original_stats': self.calculate_model_size(model),
            'compression_results': {}
        }

        # 获取压缩配置
        pruning_ratios = self.config['compression']['pruning_ratios']
        quantization_bits = self.config['compression']['quantization_bits']

        # 1. 仅剪枝
        for ratio in pruning_ratios:
            logger.info(f"测试剪枝比例: {ratio}")

            pruned_model = self.prune_model(model, ratio)
            pruned_model = self.remove_pruning_masks(pruned_model)

            eval_results = self.evaluate_compressed_model(
                model, pruned_model, sample_input, test_loader
            )

            results['compression_results'][f'pruning_{ratio}'] = eval_results

            # 保存压缩模型
            save_path = f"checkpoints/{model_name}_pruned_{ratio}.pth"
            torch.save(pruned_model.state_dict(), save_path)

        # 2. 仅量化
        for bits in quantization_bits:
            logger.info(f"测试量化位数: {bits}")

            try:
                quantized_model = self.quantize_model(model, bits)

                eval_results = self.evaluate_compressed_model(
                    model, quantized_model, sample_input, test_loader
                )

                results['compression_results'][f'quantization_{bits}bit'] = eval_results

                # 保存量化模型
                save_path = f"checkpoints/{model_name}_quantized_{bits}bit.pth"
                torch.save(quantized_model.state_dict(), save_path)

            except Exception as e:
                logger.error(f"量化失败 ({bits}bit): {str(e)}")

        # 3. 剪枝+量化组合
        for ratio in pruning_ratios[:2]:  # 只测试前两个剪枝比例
            for bits in quantization_bits[:2]:  # 只测试前两个量化位数
                logger.info(f"测试剪枝({ratio}) + 量化({bits}bit)")

                try:
                    # 先剪枝再量化
                    pruned_model = self.prune_model(model, ratio)
                    pruned_model = self.remove_pruning_masks(pruned_model)
                    combined_model = self.quantize_model(pruned_model, bits)

                    eval_results = self.evaluate_compressed_model(
                        model, combined_model, sample_input, test_loader
                    )

                    results['compression_results'][f'pruning_{ratio}_quantization_{bits}bit'] = eval_results

                    # 保存组合模型
                    save_path = f"checkpoints/{model_name}_pruned_{ratio}_quantized_{bits}bit.pth"
                    torch.save(combined_model.state_dict(), save_path)

                except Exception as e:
                    logger.error(f"组合压缩失败 (剪枝{ratio} + 量化{bits}bit): {str(e)}")

        return results

    def generate_compression_report(self, all_results: Dict[str, Dict]) -> str:
        """生成压缩报告"""
        report = []
        report.append("# 模型压缩报告\n")
        report.append(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        for model_name, results in all_results.items():
            report.append(f"## {model_name.upper()} 模型压缩结果\n")

            original_stats = results['original_stats']
            report.append(f"**原始模型统计**:\n")
            report.append(f"- 参数数量: {original_stats['total_params']:,}\n")
            report.append(f"- 模型大小: {original_stats['model_size_mb']:.2f} MB\n\n")

            # 压缩结果表格
            report.append("| 压缩方法 | 压缩比 | 参数减少 | 速度提升 | 准确率下降 |\n")
            report.append("|---------|-------|---------|---------|----------|\n")

            for method, result in results['compression_results'].items():
                size_comp = result['size_comparison']
                speed_comp = result['speed_comparison']

                compression_ratio = f"{size_comp['compression_ratio']:.2f}x"
                param_reduction = f"{size_comp['parameter_reduction']:.1%}"
                speedup = f"{speed_comp['speedup']:.2f}x"

                if 'accuracy_comparison' in result:
                    acc_drop = f"{result['accuracy_comparison']['accuracy_drop']:.2%}"
                else:
                    acc_drop = "N/A"

                report.append(f"| {method} | {compression_ratio} | {param_reduction} | {speedup} | {acc_drop} |\n")

            report.append("\n")

        # 总结和建议
        report.append("## 总结和建议\n")
        report.append("**最佳压缩策略**:\n")

        # 这里可以添加自动选择最佳压缩策略的逻辑
        report.append("- 平衡性能和压缩率: 建议使用30%剪枝 + 8位量化\n")
        report.append("- 极致压缩: 建议使用70%剪枝 + 4位量化\n")
        report.append("- 部署建议: 根据目标设备的计算能力和存储限制选择合适的压缩方案\n")

        return "\n".join(report)

    def save_compression_results(self, all_results: Dict[str, Dict]):
        """保存压缩结果"""

        # 保存详细结果
        import json
        with open('results/compression_results.json', 'w') as f:
            # 移除不能序列化的对象
            serializable_results = {}
            for model_name, results in all_results.items():
                serializable_results[model_name] = {
                    'model_name': results['model_name'],
                    'original_stats': results['original_stats'],
                    'compression_summary': {}
                }

                for method, result in results['compression_results'].items():
                    serializable_results[model_name]['compression_summary'][method] = {
                        'compression_ratio': result['size_comparison']['compression_ratio'],
                        'parameter_reduction': result['size_comparison']['parameter_reduction'],
                        'speedup': result['speed_comparison']['speedup']
                    }

            json.dump(serializable_results, f, indent=2)

        # 生成并保存报告
        report = self.generate_compression_report(all_results)
        with open('results/compression_report.md', 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info("压缩结果已保存到 results/ 目录")


def compress_models(model_paths: Dict[str, str], config: Dict) -> Dict[str, Any]:
    """
    压缩模型的入口函数
    """
    compressor = ModelCompressor(config)
    all_results = {}

    # 创建样本输入
    batch_size = 32
    seq_len = 24
    input_dim = 64

    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue

        try:
            logger.info(f"开始处理 {model_type} 模型...")

            # 加载模型
            from trainer import ModelTrainer
            trainer = ModelTrainer(config)

            if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
                input_dim = config['models']['rnn']['input_dim']
                sample_input = torch.randn(batch_size, seq_len, input_dim)
            else:  # GNN模型
                input_dim = config['models']['gnn']['input_dim']
                # 为GNN创建图数据样本
                from torch_geometric.data import Data
                num_nodes = 20
                sample_input = Data(
                    x=torch.randn(num_nodes, input_dim),
                    edge_index=torch.randint(0, num_nodes, (2, 40))
                )

            model = trainer.create_model(model_type, input_dim)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # 压缩和评估
            results = compressor.compress_and_evaluate(
                model, model_type, sample_input
            )

            all_results[model_type] = results
            logger.info(f"✅ {model_type} 模型压缩完成")

        except Exception as e:
            logger.error(f"❌ {model_type} 模型压缩失败: {str(e)}")

    # 保存结果
    if all_results:
        compressor.save_compression_results(all_results)

    return all_results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='模型压缩')
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
        results = compress_models(model_paths, config)
        logger.info("模型压缩完成!")
    else:
        logger.error("未找到模型文件")
