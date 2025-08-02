#!/usr/bin/env python3
"""
全球供应链风险预测项目主入口
基于轻量化RNN和GNN的全球供应链风险预测系统
"""

import argparse
import sys
import os
import yaml
import logging
from datetime import datetime

def setup_logging():
    """设置日志系统"""
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger(__name__)

def print_banner():
    """打印项目横幅"""
    print("=" * 80)
    print("🌐 全球供应链风险预测系统")
    print("📊 基于轻量化RNN和GNN的风险预测与部署优化")
    print("🎯 论文实验代码 - 完整工作流程")
    print("=" * 80)

def main():
    parser = argparse.ArgumentParser(
        description="全球供应链风险预测系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 运行完整流程
  python main.py --mode all
  
  # 只训练模型
  python main.py --mode train
  
  # 只评估模型
  python main.py --mode eval
  
  # 模型压缩
  python main.py --mode compress
  
  # 鲁棒性测试
  python main.py --mode robust
  
  # 部署基准测试
  python main.py --mode deploy
  
  # 超参数优化
  python main.py --mode optimize --model lstm --trials 50
        """
    )

    parser.add_argument(
        '--mode',
        choices=['all', 'train', 'eval', 'compress', 'robust', 'deploy', 'optimize'],
        default='all',
        help='运行模式'
    )

    parser.add_argument(
        '--config',
        default='config.yaml',
        help='配置文件路径'
    )

    parser.add_argument(
        '--model',
        choices=['rnn', 'lstm', 'gru', 'gcn', 'gat', 'graphsage', 'transformer'],
        help='指定模型类型（用于单模型训练或优化）'
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=50,
        help='超参数优化试验次数'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='详细输出'
    )

    args = parser.parse_args()

    # 设置日志
    logger = setup_logging()

    print_banner()

    # 检查配置文件
    if not os.path.exists(args.config):
        logger.error(f"配置文件不存在: {args.config}")
        sys.exit(1)

    # 加载配置
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"配置文件加载成功: {args.config}")
    except Exception as e:
        logger.error(f"配置文件加载失败: {e}")
        sys.exit(1)

    # 创建必要目录
    directories = ['checkpoints', 'data', 'logs', 'results']
    for dir_name in directories:
        os.makedirs(dir_name, exist_ok=True)

    logger.info(f"运行模式: {args.mode}")

    try:
        if args.mode == 'all':
            logger.info("🚀 开始完整流程执行...")
            run_full_pipeline(config, logger)

        elif args.mode == 'train':
            logger.info("🏋️ 开始模型训练...")
            run_training(config, args.model, logger)

        elif args.mode == 'eval':
            logger.info("📊 开始模型评估...")
            run_evaluation(config, logger)

        elif args.mode == 'compress':
            logger.info("🗜️ 开始模型压缩...")
            run_compression(config, logger)

        elif args.mode == 'robust':
            logger.info("🛡️ 开始鲁棒性测试...")
            run_robustness_test(config, logger)

        elif args.mode == 'deploy':
            logger.info("🚀 开始部署基准测试...")
            run_deployment_benchmark(config, logger)

        elif args.mode == 'optimize':
            if not args.model:
                logger.error("超参数优化模式需要指定模型类型")
                sys.exit(1)
            logger.info(f"🔧 开始{args.model}模型超参数优化...")
            run_hyperparameter_optimization(config, args.model, args.trials, logger)

        logger.info("✅ 所有任务执行完成！")
        print_results_summary()

    except KeyboardInterrupt:
        logger.info("❌ 用户中断执行")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 执行失败: {e}")
        sys.exit(1)

def run_full_pipeline(config, logger):
    """运行完整流程"""
    steps = [
        ("数据准备", run_data_preparation),
        ("模型训练", lambda c, l: run_training(c, None, l)),
        ("模型评估", run_evaluation),
        ("模型压缩", run_compression),
        ("鲁棒性测试", run_robustness_test),
        ("部署基准测试", run_deployment_benchmark)
    ]

    for step_name, step_func in steps:
        logger.info(f"📋 执行步骤: {step_name}")
        step_func(config, logger)
        logger.info(f"✅ {step_name} 完成")

def run_data_preparation(config, logger):
    """数据准备"""
    from data_loader import load_raw_data
    from preprocessor import preprocess
    from features import extract_features

    logger.info("加载原始数据...")
    raw_data = load_raw_data(config['data'])

    logger.info("数据预处理...")
    train_df, val_df, test_df = preprocess(raw_data, config['data'])

    logger.info("特征工程...")
    train_features = extract_features(train_df, config['features'])

    logger.info(f"数据准备完成 - 训练集特征维度: {train_features.shape}")

def run_training(config, model_type, logger):
    """运行训练"""
    from trainer import train

    if model_type:
        logger.info(f"训练单个模型: {model_type}")
        results = train(config, model_type)
    else:
        logger.info("训练所有模型...")
        results = train(config)

    logger.info("训练结果:")
    for model, result in results.items():
        if 'error' not in result:
            logger.info(f"  {model}: 验证损失 = {result['final_val_loss']:.4f}")
        else:
            logger.error(f"  {model}: 训练失败 - {result['error']}")

def run_evaluation(config, logger):
    """运行评估"""
    from evaluator import evaluate

    # 查找训练好的模型
    checkpoint_dir = config['training']['checkpoint_dir']
    model_paths = {}

    if os.path.exists(checkpoint_dir):
        for model_file in os.listdir(checkpoint_dir):
            if model_file.startswith('best_') and model_file.endswith('_model.pth'):
                model_type = model_file.replace('best_', '').replace('_model.pth', '')
                model_paths[model_type] = os.path.join(checkpoint_dir, model_file)

    if not model_paths:
        logger.warning("未找到训练好的模型，跳过评估")
        return

    logger.info(f"找到 {len(model_paths)} 个模型进行评估")
    results = evaluate(model_paths, config)

    if results:
        logger.info("评估完成，结果已保存到 results/ 目录")

def run_compression(config, logger):
    """运行模型压缩"""
    from compression import compress_models

    checkpoint_dir = config['training']['checkpoint_dir']
    model_paths = {}

    if os.path.exists(checkpoint_dir):
        for model_file in os.listdir(checkpoint_dir):
            if model_file.startswith('best_') and model_file.endswith('_model.pth'):
                model_type = model_file.replace('best_', '').replace('_model.pth', '')
                model_paths[model_type] = os.path.join(checkpoint_dir, model_file)

    if not model_paths:
        logger.warning("未找到训练好的模型，跳过压缩")
        return

    logger.info(f"对 {len(model_paths)} 个模型进行压缩")
    results = compress_models(model_paths, config)

    if results:
        logger.info("模型压缩完成，结果已保存到 results/ 目录")

def run_robustness_test(config, logger):
    """运行鲁棒性测试"""
    from robustness import run_robustness_tests

    checkpoint_dir = config['training']['checkpoint_dir']
    model_paths = {}

    if os.path.exists(checkpoint_dir):
        for model_file in os.listdir(checkpoint_dir):
            if model_file.startswith('best_') and model_file.endswith('_model.pth'):
                model_type = model_file.replace('best_', '').replace('_model.pth', '')
                model_paths[model_type] = os.path.join(checkpoint_dir, model_file)

    if not model_paths:
        logger.warning("未找到训练好的模型，跳过鲁棒性测试")
        return

    logger.info(f"对 {len(model_paths)} 个模型进行鲁棒性测试")
    results = run_robustness_tests(model_paths, config)

    if results:
        logger.info("鲁棒性测试完成，结果已保存到 results/ 目录")

def run_deployment_benchmark(config, logger):
    """运行部署基准测试"""
    from deployment import benchmark_deployment

    checkpoint_dir = config['training']['checkpoint_dir']
    model_paths = {}

    if os.path.exists(checkpoint_dir):
        for model_file in os.listdir(checkpoint_dir):
            if model_file.startswith('best_') and model_file.endswith('_model.pth'):
                model_type = model_file.replace('best_', '').replace('_model.pth', '')
                model_paths[model_type] = os.path.join(checkpoint_dir, model_file)

    if not model_paths:
        logger.warning("未找到训练好的模型，跳过部署基准测试")
        return

    logger.info(f"对 {len(model_paths)} 个模型进行部署基准测试")
    results = benchmark_deployment(model_paths, config)

    if results:
        logger.info("部署基准测试完成，结果已保存到 results/ 目录")

def run_hyperparameter_optimization(config, model_type, trials, logger):
    """运行超参数优化"""
    from trainer import hyperparameter_optimization

    logger.info(f"开始 {model_type} 模型超参数优化，试验次数: {trials}")
    results = hyperparameter_optimization(config, model_type, trials)

    logger.info("超参数优化完成:")
    logger.info(f"最佳参数: {results['best_params']}")
    logger.info(f"最佳值: {results['best_value']}")

def print_results_summary():
    """打印结果摘要"""
    print("\n" + "=" * 60)
    print("📋 结果文件摘要")
    print("=" * 60)

    result_files = [
        ("训练结果", "results/training_results.json"),
        ("评估报告", "results/evaluation_report.md"),
        ("模型比较", "results/model_comparison.csv"),
        ("压缩报告", "results/compression_report.md"),
        ("鲁棒性报告", "results/robustness_report.md"),
        ("部署报告", "results/deployment_report.md"),
        ("性能图表", "results/*.png")
    ]

    for name, path in result_files:
        if "*" in path:
            # 通配符文件，检查目录
            import glob
            files = glob.glob(path)
            if files:
                print(f"✅ {name}: {len(files)} 个文件")
            else:
                print(f"❌ {name}: 未生成")
        else:
            if os.path.exists(path):
                print(f"✅ {name}: {path}")
            else:
                print(f"❌ {name}: 未生成")

    print("=" * 60)
    print("🎉 实验完成！请查看 results/ 目录获取详细结果")
    print("=" * 60)

if __name__ == "__main__":
    main()
