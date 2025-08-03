"""
部署基准测试模块 - 边缘设备推理性能评估
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import os
import json
import time
import psutil
import platform
import subprocess
from memory_profiler import memory_usage
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class DeploymentBenchmark:
    """部署基准测试器"""

    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cpu')  # 边缘设备通常使用CPU
        self.results_dir = "results"
        os.makedirs(self.results_dir, exist_ok=True)

        # 获取系统信息
        self.system_info = self.get_system_info()
        logger.info(f"系统信息: {self.system_info}")

    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        info = {
            'platform': platform.platform(),
            'architecture': platform.architecture(),
            'processor': platform.processor(),
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            'python_version': platform.python_version(),
            'pytorch_version': torch.__version__
        }

        # 检测是否为ARM架构
        info['is_arm'] = 'arm' in platform.machine().lower() or 'aarch64' in platform.machine().lower()

        return info

    def create_sample_inputs(self, model_type: str, batch_size: int = 1) -> torch.Tensor:
        """创建样本输入数据"""

        if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
            # 序列数据
            seq_len = self.config.get('sequence_length', 24)
            input_dim = self.config['models']['rnn']['input_dim']
            sample_input = torch.randn(batch_size, seq_len, input_dim)

        elif model_type in ['gcn', 'gat', 'graphsage', 'hybrid_gnn']:
            # 图数据
            from torch_geometric.data import Data, Batch

            num_nodes = 20
            input_dim = self.config['models']['gnn']['input_dim']

            graphs = []
            for _ in range(batch_size):
                x = torch.randn(num_nodes, input_dim)
                edge_index = torch.randint(0, num_nodes, (2, 40))
                graph = Data(x=x, edge_index=edge_index)
                graphs.append(graph)

            sample_input = Batch.from_data_list(graphs) if batch_size > 1 else graphs[0]

        else:
            raise ValueError(f"未知的模型类型: {model_type}")

        return sample_input

    def benchmark_inference(self, model: nn.Module, sample_input: torch.Tensor,
                          num_warmup: int = 10, num_runs: int = 100) -> Dict[str, float]:
        """
        基准测试推理性能

        Args:
            model: 待测试的模型
            sample_input: 样本输入
            num_warmup: 预热次数
            num_runs: 测试次数

        Returns:
            性能指标字典
        """
        model.eval()
        model.to(self.device)

        if hasattr(sample_input, 'to'):
            sample_input = sample_input.to(self.device)

        # 预热
        logger.info(f"预热 {num_warmup} 次...")
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(sample_input)

        # 基准测试
        logger.info(f"基准测试 {num_runs} 次...")
        inference_times = []

        with torch.no_grad():
            for i in range(num_runs):
                start_time = time.perf_counter()
                _ = model(sample_input)
                end_time = time.perf_counter()

                inference_time = (end_time - start_time) * 1000  # 转换为毫秒
                inference_times.append(inference_time)

                if (i + 1) % 20 == 0:
                    logger.info(f"完成 {i + 1}/{num_runs} 次测试")

        # 计算统计信息
        inference_times = np.array(inference_times)

        return {
            'mean_ms': np.mean(inference_times),
            'std_ms': np.std(inference_times),
            'min_ms': np.min(inference_times),
            'max_ms': np.max(inference_times),
            'median_ms': np.median(inference_times),
            'p95_ms': np.percentile(inference_times, 95),
            'p99_ms': np.percentile(inference_times, 99),
            'throughput_fps': 1000.0 / np.mean(inference_times)  # 每秒处理帧数
        }

    def measure_memory_usage(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, float]:
        """测量内存使用情况"""

        def inference_func():
            model.eval()
            with torch.no_grad():
                _ = model(sample_input)

        # 测量推理过程中的内存使用
        logger.info("测量内存使用...")
        memory_before = psutil.virtual_memory().used / (1024**2)  # MB

        # 使用memory_profiler测量峰值内存
        peak_memory = max(memory_usage((inference_func, ())))

        memory_after = psutil.virtual_memory().used / (1024**2)  # MB

        # 计算模型参数内存
        param_memory = 0
        for param in model.parameters():
            param_memory += param.nelement() * param.element_size()

        param_memory_mb = param_memory / (1024**2)

        return {
            'peak_memory_mb': peak_memory,
            'memory_increase_mb': memory_after - memory_before,
            'parameter_memory_mb': param_memory_mb,
            'total_memory_used_mb': peak_memory,
            'memory_usage_percentage': (peak_memory / (psutil.virtual_memory().total / (1024**2))) * 100
        }

    def measure_model_size(self, model: nn.Module, model_path: str = None) -> Dict[str, Any]:
        """测量模型大小"""

        # 参数统计
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 内存大小
        param_size = 0
        buffer_size = 0

        for param in model.parameters():
            param_size += param.nelement() * param.element_size()

        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        model_size_mb = (param_size + buffer_size) / (1024**2)

        # 文件大小（如果提供了模型路径）
        file_size_mb = 0
        if model_path and os.path.exists(model_path):
            file_size_mb = os.path.getsize(model_path) / (1024**2)

        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'file_size_mb': file_size_mb,
            'parameter_density': trainable_params / total_params if total_params > 0 else 0
        }

    def cpu_utilization_test(self, model: nn.Module, sample_input: torch.Tensor,
                           duration_seconds: int = 30) -> Dict[str, float]:
        """测试CPU利用率"""

        logger.info(f"CPU利用率测试 ({duration_seconds}秒)...")

        model.eval()
        cpu_usage = []

        def inference_worker():
            """推理工作线程"""
            end_time = time.time() + duration_seconds
            with torch.no_grad():
                while time.time() < end_time:
                    _ = model(sample_input)

        # 启动推理线程
        import threading
        inference_thread = threading.Thread(target=inference_worker)

        # 监控CPU使用率
        inference_thread.start()

        start_time = time.time()
        while inference_thread.is_alive():
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_usage.append(cpu_percent)

            if time.time() - start_time > duration_seconds + 5:  # 超时保护
                break

        inference_thread.join(timeout=5)

        cpu_usage = np.array(cpu_usage)

        return {
            'avg_cpu_usage': np.mean(cpu_usage),
            'max_cpu_usage': np.max(cpu_usage),
            'min_cpu_usage': np.min(cpu_usage),
            'cpu_usage_std': np.std(cpu_usage)
        }

    def temperature_monitoring(self, model: nn.Module, sample_input: torch.Tensor,
                             duration_seconds: int = 30) -> Dict[str, Any]:
        """温度监控（如果可用）"""

        logger.info("尝试监控系统温度...")

        try:
            import concurrent.futures
            import threading
            
            def get_temperatures_safe(timeout_seconds=2):
                """安全获取温度，使用线程超时机制"""
                result = {'temps': None, 'error': None}
                
                def get_temps():
                    try:
                        result['temps'] = psutil.sensors_temperatures()
                    except Exception as e:
                        result['error'] = str(e)
                
                # 使用线程和超时
                thread = threading.Thread(target=get_temps)
                thread.daemon = True
                thread.start()
                thread.join(timeout=timeout_seconds)
                
                if thread.is_alive():
                    logger.warning(f"温度获取超时({timeout_seconds}秒)")
                    return None
                
                if result['error']:
                    logger.warning(f"温度获取错误: {result['error']}")
                    return None
                    
                return result['temps']
            
            # 快速测试温度获取（2秒超时）
            temperatures_before = get_temperatures_safe(timeout_seconds=2)
            
            if temperatures_before is None:
                logger.warning("温度传感器不可用或超时，跳过温度监控")
                return {
                    'temperature_monitoring': False,
                    'reason': 'sensors_not_available_or_timeout'
                }

            if not temperatures_before:
                logger.warning("未检测到温度传感器，跳过温度测试")
                return {
                    'temperature_monitoring': False,
                    'reason': 'no_sensors_detected'
                }

            logger.info(f"检测到温度传感器，开始温度监控测试...")
            
            # 限制最大监控时间，避免长时间运行
            actual_duration = min(duration_seconds, 5)  # 最多5秒
            logger.info(f"实际监控时间: {actual_duration}秒")
            
            # 运行推理负载
            model.eval()
            start_time = time.time()
            end_time = start_time + actual_duration
            inference_count = 0

            with torch.no_grad():
                while time.time() < end_time:
                    try:
                        _ = model(sample_input)
                        inference_count += 1
                        
                        # 每50次推理休息一下，检查时间
                        if inference_count % 50 == 0:
                            time.sleep(0.01)  # 10ms
                            current_time = time.time()
                            if current_time - start_time > actual_duration:
                                break
                            
                    except Exception as e:
                        logger.warning(f"推理过程中出错: {e}")
                        break

            logger.info(f"完成 {inference_count} 次推理，获取最终温度...")

            # 获取运行后的温度（1秒超时）
            temperatures_after = get_temperatures_safe(timeout_seconds=1)
            if temperatures_after is None:
                temperatures_after = temperatures_before

            # 分析温度变化
            temp_analysis = self._analyze_temperature_change(temperatures_before, temperatures_after)
            temp_analysis['inference_count'] = inference_count
            temp_analysis['duration_seconds'] = actual_duration

            logger.info("温度监控测试完成")
            return temp_analysis

        except Exception as e:
            logger.warning(f"温度监控失败: {str(e)}")
            return {
                'temperature_monitoring': False,
                'error': str(e)
            }

    def _analyze_temperature_change(self, temp_before: Dict, temp_after: Dict) -> Dict[str, Any]:
        """分析温度变化"""

        if not temp_before or not temp_after:
            return {'temperature_monitoring': False}

        temp_changes = []

        for sensor_name in temp_before:
            if sensor_name in temp_after:
                before_temps = [t.current for t in temp_before[sensor_name]]
                after_temps = [t.current for t in temp_after[sensor_name]]

                if before_temps and after_temps:
                    avg_before = np.mean(before_temps)
                    avg_after = np.mean(after_temps)
                    temp_changes.append(avg_after - avg_before)

        if temp_changes:
            return {
                'temperature_monitoring': True,
                'avg_temp_increase': np.mean(temp_changes),
                'max_temp_increase': np.max(temp_changes),
                'temperature_sensors': len(temp_changes)
            }
        else:
            return {'temperature_monitoring': False}

    def power_consumption_estimate(self, cpu_usage: Dict[str, float],
                                 duration_seconds: int) -> Dict[str, float]:
        """估算功耗（基于CPU使用率）"""

        # 基于不同架构的典型功耗估算
        if self.system_info['is_arm']:
            # ARM设备（如树莓派）的典型功耗
            base_power_watts = 2.0  # 基础功耗
            cpu_power_per_percent = 0.05  # 每1%CPU使用率增加的功耗
        else:
            # x86设备的典型功耗
            base_power_watts = 15.0
            cpu_power_per_percent = 0.3

        avg_cpu = cpu_usage['avg_cpu_usage']
        estimated_power = base_power_watts + (avg_cpu * cpu_power_per_percent)

        # 计算能耗
        energy_wh = (estimated_power * duration_seconds) / 3600  # 瓦时

        return {
            'estimated_power_watts': estimated_power,
            'base_power_watts': base_power_watts,
            'cpu_power_watts': avg_cpu * cpu_power_per_percent,
            'energy_consumption_wh': energy_wh,
            'energy_per_inference_mwh': (energy_wh * 1000) / max(1, duration_seconds * cpu_usage.get('inference_rate', 1))
        }

    def comprehensive_benchmark(self, model: nn.Module, model_type: str,
                              model_path: str = None) -> Dict[str, Any]:
        """综合基准测试"""

        logger.info(f"开始综合基准测试: {model_type}")

        # 创建样本输入
        sample_input = self.create_sample_inputs(model_type, batch_size=1)

        results = {
            'model_type': model_type,
            'system_info': self.system_info,
            'timestamp': pd.Timestamp.now().isoformat()
        }

        try:
            # 1. 模型大小测量
            logger.info("测量模型大小...")
            size_metrics = self.measure_model_size(model, model_path)
            results['model_size'] = size_metrics

            # 2. 推理性能测试
            logger.info("推理性能测试...")
            inference_samples = self.config['deployment']['inference_samples']
            perf_metrics = self.benchmark_inference(model, sample_input, num_runs=inference_samples)
            results['inference_performance'] = perf_metrics

            # 3. 内存使用测量
            logger.info("内存使用测量...")
            memory_metrics = self.measure_memory_usage(model, sample_input)
            results['memory_usage'] = memory_metrics

            # 4. CPU利用率测试
            logger.info("CPU利用率测试...")
            cpu_metrics = self.cpu_utilization_test(model, sample_input, duration_seconds=30)
            results['cpu_utilization'] = cpu_metrics

            # 5. 温度监控
            if self.config['deployment'].get('monitor_temperature', True):
                temp_metrics = self.temperature_monitoring(model, sample_input, duration_seconds=30)
                results['temperature'] = temp_metrics

            # 6. 功耗估算
            logger.info("功耗估算...")
            power_metrics = self.power_consumption_estimate(cpu_metrics, 30)
            results['power_consumption'] = power_metrics

            # 7. 实时性分析
            results['real_time_capability'] = self.analyze_real_time_capability(perf_metrics)

            # 8. 部署建议
            results['deployment_recommendation'] = self.generate_deployment_recommendation(results)

            logger.info(f"✅ {model_type} 基准测试完成")

        except Exception as e:
            logger.error(f"❌ {model_type} 基准测试失败: {str(e)}")
            results['error'] = str(e)

        return results

    def analyze_real_time_capability(self, perf_metrics: Dict[str, float]) -> Dict[str, Any]:
        """分析实时能力"""

        mean_latency = perf_metrics['mean_ms']
        p99_latency = perf_metrics['p99_ms']
        throughput = perf_metrics['throughput_fps']

        # 实时性等级评估
        if p99_latency < 10:
            real_time_level = "优秀"
            suitability = "适合高频实时应用"
        elif p99_latency < 50:
            real_time_level = "良好"
            suitability = "适合一般实时应用"
        elif p99_latency < 100:
            real_time_level = "一般"
            suitability = "适合低延迟要求的应用"
        elif p99_latency < 500:
            real_time_level = "较差"
            suitability = "适合批处理应用"
        else:
            real_time_level = "不适合"
            suitability = "不适合实时应用"

        return {
            'real_time_level': real_time_level,
            'suitability': suitability,
            'max_supported_frequency_hz': min(throughput, 1000.0 / p99_latency),
            'latency_budget_10ms': p99_latency <= 10,
            'latency_budget_100ms': p99_latency <= 100,
            'latency_budget_1s': p99_latency <= 1000
        }

    def generate_deployment_recommendation(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """生成部署建议"""

        recommendations = {
            'overall_score': 0,
            'recommendations': [],
            'warnings': [],
            'optimizations': []
        }

        # 分析各项指标
        if 'inference_performance' in results:
            perf = results['inference_performance']
            if perf['mean_ms'] < 50:
                recommendations['recommendations'].append("推理延迟优秀，适合实时部署")
                recommendations['overall_score'] += 25
            elif perf['mean_ms'] < 200:
                recommendations['recommendations'].append("推理延迟良好，适合准实时应用")
                recommendations['overall_score'] += 15
            else:
                recommendations['warnings'].append("推理延迟较高，可能不适合实时应用")

        if 'memory_usage' in results:
            memory = results['memory_usage']
            total_memory_gb = self.system_info['memory_total_gb']
            memory_usage_percent = (memory['peak_memory_mb'] / (total_memory_gb * 1024)) * 100

            if memory_usage_percent < 20:
                recommendations['recommendations'].append("内存使用合理")
                recommendations['overall_score'] += 25
            elif memory_usage_percent < 50:
                recommendations['recommendations'].append("内存使用可接受")
                recommendations['overall_score'] += 15
            else:
                recommendations['warnings'].append("内存使用过高，可能影响系统稳定性")
                recommendations['optimizations'].append("考虑模型压缩或量化")

        if 'model_size' in results:
            size = results['model_size']
            if size['model_size_mb'] < 10:
                recommendations['recommendations'].append("模型大小适合边缘部署")
                recommendations['overall_score'] += 25
            elif size['model_size_mb'] < 100:
                recommendations['recommendations'].append("模型大小可接受")
                recommendations['overall_score'] += 15
            else:
                recommendations['warnings'].append("模型过大，建议压缩")
                recommendations['optimizations'].append("应用剪枝和量化技术")

        if 'cpu_utilization' in results:
            cpu = results['cpu_utilization']
            if cpu['avg_cpu_usage'] < 70:
                recommendations['recommendations'].append("CPU使用率合理")
                recommendations['overall_score'] += 25
            else:
                recommendations['warnings'].append("CPU使用率过高")
                recommendations['optimizations'].append("考虑模型优化或硬件加速")

        # 部署建议等级
        if recommendations['overall_score'] >= 80:
            recommendations['deployment_level'] = "推荐部署"
        elif recommendations['overall_score'] >= 60:
            recommendations['deployment_level'] = "可以部署"
        elif recommendations['overall_score'] >= 40:
            recommendations['deployment_level'] = "需要优化后部署"
        else:
            recommendations['deployment_level'] = "不建议部署"

        return recommendations

    def create_benchmark_visualizations(self, all_results: Dict[str, Dict]):
        """创建基准测试可视化"""

        # 1. 推理延迟比较
        self.plot_inference_latency(all_results)

        # 2. 内存使用比较
        self.plot_memory_usage(all_results)

        # 3. 模型大小 vs 性能
        self.plot_size_vs_performance(all_results)

        # 4. 实时性能力雷达图
        self.plot_real_time_capability_radar(all_results)

        # 5. 部署适用性热力图
        self.plot_deployment_suitability_heatmap(all_results)

    def plot_inference_latency(self, all_results: Dict[str, Dict]):
        """绘制推理延迟比较图"""

        models = []
        mean_latencies = []
        p99_latencies = []

        for model_name, results in all_results.items():
            if 'inference_performance' in results:
                models.append(model_name)
                mean_latencies.append(results['inference_performance']['mean_ms'])
                p99_latencies.append(results['inference_performance']['p99_ms'])

        if models:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # 平均延迟
            bars1 = ax1.bar(models, mean_latencies, color='skyblue', alpha=0.7)
            ax1.set_title('Average Inference Latency')
            ax1.set_ylabel('Latency (ms)')
            ax1.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, value in zip(bars1, mean_latencies):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')

            # P99延迟
            bars2 = ax2.bar(models, p99_latencies, color='lightcoral', alpha=0.7)
            ax2.set_title('P99 Inference Latency')
            ax2.set_ylabel('Latency (ms)')
            ax2.tick_params(axis='x', rotation=45)

            # 添加数值标签
            for bar, value in zip(bars2, p99_latencies):
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'inference_latency_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_memory_usage(self, all_results: Dict[str, Dict]):
        """绘制内存使用比较图"""

        models = []
        peak_memory = []
        param_memory = []

        for model_name, results in all_results.items():
            if 'memory_usage' in results and 'model_size' in results:
                models.append(model_name)
                peak_memory.append(results['memory_usage']['peak_memory_mb'])
                param_memory.append(results['model_size']['model_size_mb'])

        if models:
            fig, ax = plt.subplots(figsize=(12, 6))

            x = np.arange(len(models))
            width = 0.35

            bars1 = ax.bar(x - width/2, peak_memory, width, label='Peak Memory', alpha=0.7)
            bars2 = ax.bar(x + width/2, param_memory, width, label='Parameter Memory', alpha=0.7)

            ax.set_title('Memory Usage Comparison')
            ax.set_ylabel('Memory (MB)')
            ax.set_xticks(x)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()

            # 添加数值标签
            for bars in [bars1, bars2]:
                for bar in bars:
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                           f'{height:.1f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'memory_usage_comparison.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_size_vs_performance(self, all_results: Dict[str, Dict]):
        """绘制模型大小 vs 性能散点图"""

        sizes = []
        latencies = []
        model_names = []

        for model_name, results in all_results.items():
            if 'model_size' in results and 'inference_performance' in results:
                sizes.append(results['model_size']['model_size_mb'])
                latencies.append(results['inference_performance']['mean_ms'])
                model_names.append(model_name)

        if sizes:
            plt.figure(figsize=(10, 8))

            colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

            for i, (size, latency, name) in enumerate(zip(sizes, latencies, model_names)):
                plt.scatter(size, latency, s=100, c=[colors[i]], alpha=0.7, label=name)
                plt.annotate(name, (size, latency), xytext=(5, 5),
                           textcoords='offset points', fontsize=9)

            plt.xlabel('Model Size (MB)')
            plt.ylabel('Inference Latency (ms)')
            plt.title('Model Size vs Inference Latency')
            plt.grid(True, alpha=0.3)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'size_vs_performance.png'),
                       dpi=300, bbox_inches='tight')
            plt.close()

    def plot_real_time_capability_radar(self, all_results: Dict[str, Dict]):
        """绘制实时能力雷达图"""

        # 定义评估维度
        dimensions = ['延迟性能', '内存效率', 'CPU效率', '模型大小', '实时能力']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        angles = np.linspace(0, 2 * np.pi, len(dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 闭合图形

        colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))

        for i, (model_name, results) in enumerate(all_results.items()):
            if 'error' in results:
                continue

            scores = []

            # 延迟性能 (越小越好，归一化到0-1)
            if 'inference_performance' in results:
                latency = results['inference_performance']['mean_ms']
                latency_score = max(0, 1 - latency / 1000)  # 1秒以内得分较高
                scores.append(latency_score)
            else:
                scores.append(0)

            # 内存效率
            if 'memory_usage' in results:
                memory_mb = results['memory_usage']['peak_memory_mb']
                memory_score = max(0, 1 - memory_mb / 1000)  # 1GB以内得分较高
                scores.append(memory_score)
            else:
                scores.append(0)

            # CPU效率
            if 'cpu_utilization' in results:
                cpu_usage = results['cpu_utilization']['avg_cpu_usage']
                cpu_score = max(0, 1 - cpu_usage / 100)
                scores.append(cpu_score)
            else:
                scores.append(0)

            # 模型大小
            if 'model_size' in results:
                size_mb = results['model_size']['model_size_mb']
                size_score = max(0, 1 - size_mb / 500)  # 500MB以内得分较高
                scores.append(size_score)
            else:
                scores.append(0)

            # 实时能力
            if 'real_time_capability' in results:
                rt_level = results['real_time_capability']['real_time_level']
                rt_score_map = {'优秀': 1.0, '良好': 0.8, '一般': 0.6, '较差': 0.4, '不适合': 0.2}
                rt_score = rt_score_map.get(rt_level, 0)
                scores.append(rt_score)
            else:
                scores.append(0)

            scores += scores[:1]  # 闭合图形

            ax.plot(angles, scores, 'o-', linewidth=2, label=model_name, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(dimensions)
        ax.set_ylim(0, 1)
        ax.set_title('Real-time Deployment Capability Assessment', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'real_time_capability_radar.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def plot_deployment_suitability_heatmap(self, all_results: Dict[str, Dict]):
        """绘制部署适用性热力图"""

        # 定义部署场景
        scenarios = ['实时预警', '批量分析', '边缘计算', '云端部署', '移动设备']
        models = list(all_results.keys())

        # 创建适用性矩阵
        suitability_matrix = np.zeros((len(models), len(scenarios)))

        for i, model_name in enumerate(models):
            results = all_results[model_name]

            if 'error' in results:
                continue

            # 基于各项指标计算对不同场景的适用性
            latency = results.get('inference_performance', {}).get('mean_ms', 1000)
            memory = results.get('memory_usage', {}).get('peak_memory_mb', 1000)
            size = results.get('model_size', {}).get('model_size_mb', 1000)

            # 实时预警：需要低延迟
            suitability_matrix[i, 0] = 1.0 if latency < 100 else 0.5 if latency < 500 else 0.1

            # 批量分析：对延迟要求不高，但需要高准确性
            suitability_matrix[i, 1] = 0.8  # 假设所有模型都适合批量分析

            # 边缘计算：需要小模型和低内存
            edge_score = 1.0
            if size > 50:
                edge_score *= 0.7
            if memory > 200:
                edge_score *= 0.7
            suitability_matrix[i, 2] = edge_score

            # 云端部署：资源充足，主要看准确性
            suitability_matrix[i, 3] = 0.9  # 假设云端适合所有模型

            # 移动设备：严格的大小和内存限制
            mobile_score = 1.0
            if size > 20:
                mobile_score *= 0.5
            if memory > 100:
                mobile_score *= 0.5
            if latency > 200:
                mobile_score *= 0.7
            suitability_matrix[i, 4] = mobile_score

        # 绘制热力图
        plt.figure(figsize=(10, 8))
        sns.heatmap(suitability_matrix,
                   xticklabels=scenarios,
                   yticklabels=models,
                   annot=True,
                   fmt='.2f',
                   cmap='RdYlGn',
                   cbar_kws={'label': 'Suitability Score'})

        plt.title('Model Deployment Scenario Suitability')
        plt.xlabel('Deployment Scenario')
        plt.ylabel('Model')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'deployment_suitability_heatmap.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()

    def generate_benchmark_report(self, all_results: Dict[str, Dict]) -> str:
        """生成基准测试报告"""

        report = []
        report.append("# 边缘设备部署基准测试报告\n")
        report.append(f"测试时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # 系统信息
        report.append("## 测试环境\n")
        system_info = self.system_info
        report.append(f"- **平台**: {system_info['platform']}\n")
        report.append(f"- **架构**: {system_info['architecture'][0]}\n")
        report.append(f"- **处理器**: {system_info['processor']}\n")
        report.append(f"- **CPU核心数**: {system_info['cpu_count']}\n")
        report.append(f"- **内存**: {system_info['memory_total_gb']} GB\n")
        report.append(f"- **是否ARM**: {'是' if system_info['is_arm'] else '否'}\n\n")

        # 性能总结表
        report.append("## 性能总结\n")
        report.append("| 模型 | 平均延迟(ms) | P99延迟(ms) | 峰值内存(MB) | 模型大小(MB) | 实时等级 |\n")
        report.append("|------|-------------|------------|-------------|-------------|----------|\n")

        for model_name, results in all_results.items():
            if 'error' in results:
                report.append(f"| {model_name} | 错误 | 错误 | 错误 | 错误 | 错误 |\n")
                continue

            avg_latency = results.get('inference_performance', {}).get('mean_ms', 'N/A')
            p99_latency = results.get('inference_performance', {}).get('p99_ms', 'N/A')
            peak_memory = results.get('memory_usage', {}).get('peak_memory_mb', 'N/A')
            model_size = results.get('model_size', {}).get('model_size_mb', 'N/A')
            rt_level = results.get('real_time_capability', {}).get('real_time_level', 'N/A')

            report.append(f"| {model_name} | {avg_latency:.1f} | {p99_latency:.1f} | {peak_memory:.1f} | {model_size:.1f} | {rt_level} |\n")

        report.append("\n")

        # 详细分析
        report.append("## 详细分析\n")

        for model_name, results in all_results.items():
            if 'error' in results:
                report.append(f"### {model_name.upper()} - 测试失败\n")
                report.append(f"错误信息: {results['error']}\n\n")
                continue

            report.append(f"### {model_name.upper()}\n")

            # 性能指标
            if 'inference_performance' in results:
                perf = results['inference_performance']
                report.append(f"**推理性能**:\n")
                report.append(f"- 平均延迟: {perf['mean_ms']:.2f} ms\n")
                report.append(f"- P95延迟: {perf['p95_ms']:.2f} ms\n")
                report.append(f"- P99延迟: {perf['p99_ms']:.2f} ms\n")
                report.append(f"- 吞吐量: {perf['throughput_fps']:.2f} FPS\n\n")

            # 资源使用
            if 'memory_usage' in results:
                memory = results['memory_usage']
                report.append(f"**内存使用**:\n")
                report.append(f"- 峰值内存: {memory['peak_memory_mb']:.2f} MB\n")
                report.append(f"- 参数内存: {memory['parameter_memory_mb']:.2f} MB\n")
                report.append(f"- 内存占用率: {memory['memory_usage_percentage']:.1f}%\n\n")

            # 部署建议
            if 'deployment_recommendation' in results:
                rec = results['deployment_recommendation']
                report.append(f"**部署建议**:\n")
                report.append(f"- 综合评分: {rec['overall_score']}/100\n")
                report.append(f"- 部署等级: {rec['deployment_level']}\n")

                if rec['recommendations']:
                    report.append("- 优势:\n")
                    for r in rec['recommendations']:
                        report.append(f"  - {r}\n")

                if rec['warnings']:
                    report.append("- 注意事项:\n")
                    for w in rec['warnings']:
                        report.append(f"  - {w}\n")

                if rec['optimizations']:
                    report.append("- 优化建议:\n")
                    for o in rec['optimizations']:
                        report.append(f"  - {o}\n")

                report.append("\n")

        # 总结和建议
        report.append("## 总结和建议\n")

        # 找出最佳模型
        best_models = {}
        metrics = ['inference_performance', 'memory_usage', 'model_size']

        for metric in metrics:
            best_score = float('inf') if metric != 'inference_performance' else 0
            best_model = None

            for model_name, results in all_results.items():
                if 'error' in results or metric not in results:
                    continue

                if metric == 'inference_performance':
                    score = results[metric]['throughput_fps']
                    if score > best_score:
                        best_score = score
                        best_model = model_name
                elif metric == 'memory_usage':
                    score = results[metric]['peak_memory_mb']
                    if score < best_score:
                        best_score = score
                        best_model = model_name
                elif metric == 'model_size':
                    score = results[metric]['model_size_mb']
                    if score < best_score:
                        best_score = score
                        best_model = model_name

            if best_model:
                best_models[metric] = best_model

        report.append("**各项最佳模型**:\n")
        metric_names = {
            'inference_performance': '推理性能',
            'memory_usage': '内存效率',
            'model_size': '模型大小'
        }

        for metric, model in best_models.items():
            report.append(f"- {metric_names[metric]}: {model}\n")

        report.append("\n**部署场景推荐**:\n")
        report.append("- **实时预警系统**: 选择延迟最低的模型\n")
        report.append("- **边缘计算设备**: 选择模型大小和内存使用最优的模型\n")
        report.append("- **批量处理**: 可选择准确性最高的模型，延迟要求相对宽松\n")
        report.append("- **移动应用**: 严格选择小模型，并考虑量化压缩\n")

        return "\n".join(report)

    def save_benchmark_results(self, all_results: Dict[str, Dict]):
        """保存基准测试结果"""

        # 保存详细结果
        with open(os.path.join(self.results_dir, 'deployment_benchmark.json'), 'w') as f:
            json.dump(all_results, f, indent=2, default=str)

        # 生成CSV摘要
        summary_data = []
        for model_name, results in all_results.items():
            if 'error' in results:
                continue

            row = {'model': model_name}

            if 'inference_performance' in results:
                perf = results['inference_performance']
                row.update({
                    'mean_latency_ms': perf['mean_ms'],
                    'p99_latency_ms': perf['p99_ms'],
                    'throughput_fps': perf['throughput_fps']
                })

            if 'memory_usage' in results:
                memory = results['memory_usage']
                row.update({
                    'peak_memory_mb': memory['peak_memory_mb'],
                    'parameter_memory_mb': memory['parameter_memory_mb']
                })

            if 'model_size' in results:
                size = results['model_size']
                row.update({
                    'model_size_mb': size['model_size_mb'],
                    'total_parameters': size['total_parameters']
                })

            if 'deployment_recommendation' in results:
                rec = results['deployment_recommendation']
                row.update({
                    'deployment_score': rec['overall_score'],
                    'deployment_level': rec['deployment_level']
                })

            summary_data.append(row)

        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(os.path.join(self.results_dir, 'deployment_summary.csv'), index=False)

        # 生成并保存报告
        report = self.generate_benchmark_report(all_results)
        with open(os.path.join(self.results_dir, 'deployment_report.md'), 'w', encoding='utf-8') as f:
            f.write(report)

        logger.info("部署基准测试结果已保存到 results/ 目录")


def benchmark_deployment(model_paths: Dict[str, str], config: Dict) -> Dict[str, Any]:
    """
    部署基准测试入口函数
    """

    benchmark = DeploymentBenchmark(config)
    all_results = {}

    for model_type, model_path in model_paths.items():
        if not os.path.exists(model_path):
            logger.warning(f"模型文件不存在: {model_path}")
            continue

        try:
            logger.info(f"开始基准测试: {model_type}")

            # 加载模型
            from trainer import ModelTrainer
            trainer = ModelTrainer(config)

            if model_type in ['rnn', 'lstm', 'gru', 'transformer']:
                input_dim = config['models']['rnn']['input_dim']
            else:
                input_dim = config['models']['gnn']['input_dim']

            model = trainer.create_model(model_type, input_dim)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            
            # 处理不同的保存格式
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)

            # 综合基准测试
            results = benchmark.comprehensive_benchmark(model, model_type, model_path)
            all_results[model_type] = results

        except Exception as e:
            logger.error(f"❌ {model_type} 基准测试失败: {str(e)}")
            all_results[model_type] = {'error': str(e)}

    if all_results:
        # 创建可视化
        benchmark.create_benchmark_visualizations(all_results)

        # 保存结果
        benchmark.save_benchmark_results(all_results)

    return all_results


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser(description='部署基准测试')
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
        results = benchmark_deployment(model_paths, config)
        logger.info("部署基准测试完成!")
    else:
        logger.error("未找到模型文件")
