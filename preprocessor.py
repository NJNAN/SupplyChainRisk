"""
数据预处理模块 - 时间对齐、缺失值处理、数据划分
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


def preprocess(raw_data: Dict[str, pd.DataFrame], config: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    数据预处理主函数

    Args:
        raw_data: 包含原始数据的字典
        config: 配置参数

    Returns:
        训练集、验证集、测试集的元组
    """
    logger.info("开始数据预处理...")

    # 合并所有数据源
    combined_df = merge_data_sources(raw_data, config)

    # 时间对齐和时区处理
    combined_df = align_timestamps(combined_df, config)

    # 缺失值处理
    combined_df = handle_missing_values(combined_df, config)

    # 数据质量检查
    combined_df = quality_check(combined_df)

    # 时间序列划分
    train_df, val_df, test_df = temporal_split(combined_df, config)

    # 确保所有子集包含所有类别
    train_df, val_df, test_df = ensure_category_coverage(train_df, val_df, test_df)

    logger.info(f"预处理完成 - 训练集: {train_df.shape}, 验证集: {val_df.shape}, 测试集: {test_df.shape}")

    return train_df, val_df, test_df


def merge_data_sources(raw_data: Dict[str, pd.DataFrame], config: Dict) -> pd.DataFrame:
    """
    合并多个数据源
    """
    dfs_to_merge = []

    # 处理Comtrade数据
    if 'comtrade' in raw_data and not raw_data['comtrade'].empty:
        comtrade_df = raw_data['comtrade'].copy()
        comtrade_df['data_source'] = 'comtrade'
        # 标准化列名
        if 'delay_flag' in comtrade_df.columns:
            comtrade_df['target'] = comtrade_df['delay_flag']
        dfs_to_merge.append(comtrade_df)

    # 处理AIS数据
    if 'ais' in raw_data and not raw_data['ais'].empty:
        ais_df = raw_data['ais'].copy()
        ais_df['data_source'] = 'ais'
        if 'anomaly_flag' in ais_df.columns:
            ais_df['target'] = ais_df['anomaly_flag']
        dfs_to_merge.append(ais_df)

    # 处理合成数据
    if 'synthetic' in raw_data and not raw_data['synthetic'].empty:
        synthetic_df = raw_data['synthetic'].copy()
        synthetic_df['data_source'] = 'synthetic'
        if 'risk_flag' in synthetic_df.columns:
            synthetic_df['target'] = synthetic_df['risk_flag']
        dfs_to_merge.append(synthetic_df)

    if not dfs_to_merge:
        raise ValueError("没有可用的数据源")

    # 合并数据
    combined_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False)

    # 确保target列存在
    if 'target' not in combined_df.columns:
        combined_df['target'] = 0

    return combined_df


def align_timestamps(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    时间对齐和时区处理
    """
    time_column = config.get('time_column', 'timestamp')

    if time_column not in df.columns:
        logger.warning(f"时间列 {time_column} 不存在，使用默认时间")
        df[time_column] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

    # 转换为统一时间格式
    df[time_column] = pd.to_datetime(df[time_column], errors='coerce')

    # 删除时间转换失败的行
    before_count = len(df)
    df = df.dropna(subset=[time_column])
    after_count = len(df)

    if before_count > after_count:
        logger.warning(f"删除了 {before_count - after_count} 行时间格式异常的数据")

    # 转换为UTC时区
    if df[time_column].dt.tz is None:
        df[time_column] = df[time_column].dt.tz_localize('UTC')
    else:
        df[time_column] = df[time_column].dt.tz_convert('UTC')

    # 按时间排序
    df = df.sort_values(time_column).reset_index(drop=True)

    return df


def handle_missing_values(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    处理缺失值
    """
    strategy = config.get('missing_strategy', 'interpolate')

    logger.info(f"使用策略 '{strategy}' 处理缺失值")

    # 统计缺失值
    missing_stats = df.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0]

    if len(missing_stats) > 0:
        logger.info(f"发现缺失值列: {dict(missing_stats)}")

    if strategy == 'drop':
        # 删除含有缺失值的行
        before_count = len(df)
        df = df.dropna()
        logger.info(f"删除缺失值行: {before_count} -> {len(df)}")

    elif strategy == 'interpolate':
        # 插值填充
        numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        for col in numeric_columns:
            if df[col].isnull().any():
                df[col] = df[col].interpolate(method='linear').fillna(df[col].mean())

        # 分类变量用众数填充
        categorical_columns = df.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')

    elif strategy == 'flag':
        # 标记缺失值并填充
        for col in df.columns:
            if df[col].isnull().any():
                df[f'{col}_missing'] = df[col].isnull().astype(int)
                if df[col].dtype in ['float64', 'int64']:
                    df[col] = df[col].fillna(df[col].mean())
                else:
                    df[col] = df[col].fillna('Missing')

    return df


def quality_check(df: pd.DataFrame) -> pd.DataFrame:
    """
    数据质量检查和异常值处理
    """
    original_count = len(df)

    # 删除重复行
    df = df.drop_duplicates()

    # 处理数值型异常值（使用IQR方法）
    numeric_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    numeric_columns = [col for col in numeric_columns if col != 'target']  # 不处理目标变量

    for col in numeric_columns:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            # 定义异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            # 将异常值替换为边界值（而不是删除）
            df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)

    logger.info(f"数据质量检查完成: {original_count} -> {len(df)} 行")

    return df


def temporal_split(df: pd.DataFrame, config: Dict,
                  train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    按时间顺序划分数据集，避免数据泄漏
    """
    time_column = config.get('time_column', 'timestamp')

    # 按时间排序
    df = df.sort_values(time_column).reset_index(drop=True)

    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    # 划分数据
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    logger.info(f"时间划分完成:")
    logger.info(f"  训练集: {train_df[time_column].min()} - {train_df[time_column].max()} ({len(train_df)} 行)")
    logger.info(f"  验证集: {val_df[time_column].min()} - {val_df[time_column].max()} ({len(val_df)} 行)")
    logger.info(f"  测试集: {test_df[time_column].min()} - {test_df[time_column].max()} ({len(test_df)} 行)")

    return train_df, val_df, test_df


def ensure_category_coverage(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    确保所有数据集都包含主要类别
    """
    # 检查目标变量分布
    if 'target' in train_df.columns:
        train_target_dist = train_df['target'].value_counts()
        val_target_dist = val_df['target'].value_counts()
        test_target_dist = test_df['target'].value_counts()

        logger.info("目标变量分布:")
        logger.info(f"  训练集: {dict(train_target_dist)}")
        logger.info(f"  验证集: {dict(val_target_dist)}")
        logger.info(f"  测试集: {dict(test_target_dist)}")

        # 检查是否有类别缺失
        all_classes = set(train_df['target'].unique())
        val_classes = set(val_df['target'].unique())
        test_classes = set(test_df['target'].unique())

        if not val_classes.issuperset(all_classes):
            logger.warning("验证集缺少某些类别")
        if not test_classes.issuperset(all_classes):
            logger.warning("测试集缺少某些类别")

    # 检查分类特征覆盖度
    categorical_columns = []
    for col in train_df.columns:
        if train_df[col].dtype == 'object' and col not in ['timestamp']:
            categorical_columns.append(col)

    for col in categorical_columns[:3]:  # 只检查前3个分类列
        if col in train_df.columns and col in val_df.columns and col in test_df.columns:
            train_cats = set(train_df[col].unique())
            val_cats = set(val_df[col].unique())
            test_cats = set(test_df[col].unique())

            logger.info(f"'{col}' 类别覆盖度:")
            logger.info(f"  训练集: {len(train_cats)}, 验证集: {len(val_cats)}, 测试集: {len(test_cats)}")

    return train_df, val_df, test_df


def create_sequences(df: pd.DataFrame, sequence_length: int = 24, target_column: str = 'target') -> Tuple[np.ndarray, np.ndarray]:
    """
    为序列模型创建时间序列数据
    """
    # 选择数值型特征
    feature_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    feature_columns = [col for col in feature_columns if col != target_column]

    X_sequences = []
    y_sequences = []

    for i in range(len(df) - sequence_length + 1):
        # 特征序列
        X_seq = df[feature_columns].iloc[i:i + sequence_length].values
        # 目标值（预测序列最后一个时间点的值）
        y_seq = df[target_column].iloc[i + sequence_length - 1]

        X_sequences.append(X_seq)
        y_sequences.append(y_seq)

    return np.array(X_sequences), np.array(y_sequences)


def create_sequences_from_features(X: np.ndarray, y: np.ndarray, sequence_length: int = 24) -> Tuple[np.ndarray, np.ndarray]:
    """
    直接从特征矩阵和目标向量创建时间序列数据
    
    Args:
        X: 特征矩阵 [n_samples, n_features]
        y: 目标向量 [n_samples]
        sequence_length: 序列长度
        
    Returns:
        X_sequences: [n_sequences, sequence_length, n_features]
        y_sequences: [n_sequences]
    """
    if len(X) < sequence_length:
        logger.warning(f"数据长度 {len(X)} 小于序列长度 {sequence_length}，返回空序列")
        return np.array([]), np.array([])
    
    X_sequences = []
    y_sequences = []
    
    for i in range(len(X) - sequence_length + 1):
        # 特征序列
        X_seq = X[i:i + sequence_length]
        # 目标值（预测序列最后一个时间点的值）
        y_seq = y[i + sequence_length - 1]
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
    
    return np.array(X_sequences), np.array(y_sequences)


def standardize_features(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict]:
    """
    标准化数值特征
    """
    # 选择数值型特征进行标准化
    numeric_columns = train_df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    numeric_columns = [col for col in numeric_columns if col != 'target']

    scalers = {}

    for col in numeric_columns:
        if col in train_df.columns:
            scaler = StandardScaler()

            # 在训练集上拟合
            train_df[col] = scaler.fit_transform(train_df[[col]])

            # 应用到验证集和测试集
            val_df[col] = scaler.transform(val_df[[col]])
            test_df[col] = scaler.transform(test_df[[col]])

            scalers[col] = scaler

    logger.info(f"标准化了 {len(scalers)} 个数值特征")

    return train_df, val_df, test_df, scalers


if __name__ == "__main__":
    # 测试代码
    import yaml
    from data_loader import load_raw_data

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载数据
    raw_data = load_raw_data(config['data'])

    # 预处理
    train_df, val_df, test_df = preprocess(raw_data, config['data'])

    print("\n预处理结果:")
    print(f"训练集形状: {train_df.shape}")
    print(f"验证集形状: {val_df.shape}")
    print(f"测试集形状: {test_df.shape}")

    # 标准化
    train_df, val_df, test_df, scalers = standardize_features(train_df, val_df, test_df)
    print(f"标准化完成，使用了 {len(scalers)} 个标准化器")
