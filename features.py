"""
特征工程模块 - 节假日编码、HS Code处理、外部数据融合
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
import os
from datetime import datetime, timedelta
import holidays

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


def extract_features(df: pd.DataFrame, config: Dict) -> np.ndarray:
    """
    特征工程主函数

    Args:
        df: 输入数据框
        config: 配置字典

    Returns:
        特征矩阵 (N, D)
    """
    logger.info("开始特征工程...")

    df_features = df.copy()

    # 时间特征工程
    df_features = add_temporal_features(df_features, config)

    # 节假日特征
    df_features = add_holiday_features(df_features, config)

    # HS Code特征处理
    df_features = process_hs_codes(df_features, config)

    # 货值特征处理
    df_features = process_trade_values(df_features, config)

    # 加载外部数据特征
    df_features = add_external_features(df_features, config)

    # 地理位置特征
    df_features = add_geographic_features(df_features)

    # 交互特征
    df_features = add_interaction_features(df_features)

    # 统计特征
    df_features = add_statistical_features(df_features)

    # 转换为数值矩阵
    feature_matrix = prepare_feature_matrix(df_features)

    logger.info(f"特征工程完成，输出特征维度: {feature_matrix.shape}")

    return feature_matrix


def add_temporal_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    添加时间特征
    """
    time_column = config.get('time_column', 'timestamp')

    if time_column in df.columns:
        dt = pd.to_datetime(df[time_column])

        # 基础时间特征
        df['year'] = dt.dt.year
        df['month'] = dt.dt.month
        df['day'] = dt.dt.day
        df['hour'] = dt.dt.hour
        df['dayofweek'] = dt.dt.dayofweek
        df['dayofyear'] = dt.dt.dayofyear
        df['quarter'] = dt.dt.quarter

        # 周期性编码（正弦余弦变换）
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)

        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 是否工作日
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 时间段标记
        df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
        df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
        df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 24)).astype(int)
        df['is_night'] = ((df['hour'] >= 0) & (df['hour'] < 6)).astype(int)

        logger.info("添加了时间特征")

    return df


def add_holiday_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    添加节假日特征
    """
    time_column = config.get('time_column', 'timestamp')
    holiday_calendar = config.get('holiday_calendar', 'CN')

    if time_column in df.columns:
        dt = pd.to_datetime(df[time_column])

        # 获取节假日日历
        if holiday_calendar == 'CN':
            cn_holidays = holidays.China()
            df['is_holiday'] = dt.dt.date.apply(lambda x: x in cn_holidays).astype(int)
        elif holiday_calendar == 'US':
            us_holidays = holidays.UnitedStates()
            df['is_holiday'] = dt.dt.date.apply(lambda x: x in us_holidays).astype(int)
        else:
            # 默认使用简单的节假日判断
            df['is_holiday'] = 0

        # 节假日前后标记
        df['holiday_lag_1'] = df['is_holiday'].shift(1).fillna(0).astype(int)
        df['holiday_lead_1'] = df['is_holiday'].shift(-1).fillna(0).astype(int)

        # 春节期间（中国特有）
        if holiday_calendar == 'CN':
            spring_festival_months = [1, 2]
            df['is_spring_festival_period'] = df['month'].isin(spring_festival_months).astype(int)
        else:
            df['is_spring_festival_period'] = 0

        logger.info(f"添加了节假日特征 (日历: {holiday_calendar})")

    return df


def process_hs_codes(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    处理HS编码特征
    """
    encoding_method = config.get('hs_code_encoding', 'onehot')

    # 寻找HS Code相关列
    hs_columns = [col for col in df.columns if 'hs' in col.lower() or 'code' in col.lower()]

    if not hs_columns:
        # 如果没有HS Code，创建一个模拟的
        df['hs_code'] = np.random.choice(['01', '02', '03', '84', '85'], len(df))
        hs_columns = ['hs_code']

    for col in hs_columns:
        if col in df.columns:
            # 清理HS代码
            df[col] = df[col].astype(str).str.upper().str.strip()

            if encoding_method == 'onehot':
                # One-Hot编码
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])

                # 创建列名
                feature_names = [f'{col}_onehot_{i}' for i in range(encoded.shape[1])]

                # 添加到数据框
                for i, feature_name in enumerate(feature_names):
                    df[feature_name] = encoded[:, i]

            elif encoding_method == 'embedding':
                # 标签编码（为嵌入做准备）
                encoder = LabelEncoder()
                df[f'{col}_encoded'] = encoder.fit_transform(df[col])

                # 添加嵌入维度信息
                df[f'{col}_vocab_size'] = len(encoder.classes_)

    logger.info(f"处理了HS Code特征，使用方法: {encoding_method}")

    return df


def process_trade_values(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    处理贸易货值特征
    """
    normalization = config.get('normalization', 'log')

    # 寻找货值相关列
    value_columns = []
    for col in df.columns:
        if any(keyword in col.lower() for keyword in ['value', 'price', 'amount', 'cost']):
            if df[col].dtype in ['float64', 'int64']:
                value_columns.append(col)

    for col in value_columns:
        if col in df.columns:
            # 确保值为正数
            df[col] = df[col].clip(lower=0.01)  # 避免log(0)

            if normalization == 'log':
                df[f'{col}_log'] = np.log1p(df[col])
            elif normalization == 'sqrt':
                df[f'{col}_sqrt'] = np.sqrt(df[col])
            elif normalization == 'boxcox':
                from scipy.stats import boxcox
                try:
                    df[f'{col}_boxcox'], _ = boxcox(df[col] + 1)
                except:
                    df[f'{col}_boxcox'] = np.log1p(df[col])  # 备选方案

            # 分位数特征
            df[f'{col}_quantile'] = df[col].rank(pct=True)

            # 异常值标记
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            df[f'{col}_outlier'] = ((df[col] < Q1 - 1.5 * IQR) |
                                   (df[col] > Q3 + 1.5 * IQR)).astype(int)

    logger.info(f"处理了 {len(value_columns)} 个货值特征")

    return df


def add_external_features(df: pd.DataFrame, config: Dict) -> pd.DataFrame:
    """
    加载并合并外部数据特征
    """
    external_data = config.get('external_data', {})

    # 港口拥堵指数
    if 'port_congestion' in external_data:
        congestion_file = external_data['port_congestion']
        if os.path.exists(congestion_file):
            try:
                congestion_df = pd.read_csv(congestion_file)
                df = merge_external_data(df, congestion_df, 'port_congestion')
            except:
                logger.warning("港口拥堵数据加载失败，使用模拟数据")
                df = add_mock_congestion_data(df)
        else:
            df = add_mock_congestion_data(df)
    else:
        df = add_mock_congestion_data(df)

    # 汇率数据
    if 'exchange_rates' in external_data:
        exchange_file = external_data['exchange_rates']
        if os.path.exists(exchange_file):
            try:
                exchange_df = pd.read_csv(exchange_file)
                df = merge_external_data(df, exchange_df, 'exchange_rate')
            except:
                logger.warning("汇率数据加载失败，使用模拟数据")
                df = add_mock_exchange_data(df)
        else:
            df = add_mock_exchange_data(df)
    else:
        df = add_mock_exchange_data(df)

    # 天气数据
    if 'weather' in external_data:
        weather_file = external_data['weather']
        if os.path.exists(weather_file):
            try:
                weather_df = pd.read_csv(weather_file)
                df = merge_external_data(df, weather_df, 'weather')
            except:
                logger.warning("天气数据加载失败，使用模拟数据")
                df = add_mock_weather_data(df)
        else:
            df = add_mock_weather_data(df)
    else:
        df = add_mock_weather_data(df)

    return df


def add_mock_congestion_data(df: pd.DataFrame) -> pd.DataFrame:
    """添加模拟港口拥堵数据"""
    df['port_congestion_index'] = np.random.exponential(0.3, len(df))
    df['port_congestion_high'] = (df['port_congestion_index'] > 0.5).astype(int)
    return df


def add_mock_exchange_data(df: pd.DataFrame) -> pd.DataFrame:
    """添加模拟汇率数据"""
    base_rate = 6.5  # USD/CNY基准汇率
    df['exchange_rate_usd_cny'] = base_rate + np.random.normal(0, 0.1, len(df))
    df['exchange_rate_volatility'] = abs(np.random.normal(0, 0.05, len(df)))
    return df


def add_mock_weather_data(df: pd.DataFrame) -> pd.DataFrame:
    """添加模拟天气数据"""
    df['weather_score'] = np.random.beta(2, 2, len(df))  # 0-1之间的天气评分
    df['weather_risk'] = (df['weather_score'] < 0.3).astype(int)
    df['temperature'] = np.random.normal(20, 10, len(df))
    df['humidity'] = np.random.uniform(30, 90, len(df))
    df['wind_speed'] = np.random.exponential(5, len(df))
    return df


def merge_external_data(df: pd.DataFrame, external_df: pd.DataFrame, data_type: str) -> pd.DataFrame:
    """合并外部数据的通用函数"""
    try:
        # 这里假设外部数据有时间列，按时间合并
        if 'timestamp' in external_df.columns:
            external_df['timestamp'] = pd.to_datetime(external_df['timestamp'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])

            # 按最近时间合并
            df = pd.merge_asof(df.sort_values('timestamp'),
                              external_df.sort_values('timestamp'),
                              on='timestamp',
                              direction='backward',
                              suffixes=('', f'_{data_type}'))

        logger.info(f"成功合并 {data_type} 外部数据")
    except Exception as e:
        logger.warning(f"{data_type} 数据合并失败: {str(e)}")

    return df


def add_geographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加地理位置特征
    """
    # 如果有经纬度信息
    if 'latitude' in df.columns and 'longitude' in df.columns:
        # 计算到主要港口的距离
        major_ports = {
            'Shanghai': (31.2304, 121.4737),
            'Singapore': (1.3521, 103.8198),
            'Rotterdam': (51.9225, 4.4792),
            'Los Angeles': (33.7357, -118.2640)
        }

        for port_name, (port_lat, port_lon) in major_ports.items():
            df[f'distance_to_{port_name.lower()}'] = haversine_distance(
                df['latitude'], df['longitude'], port_lat, port_lon
            )

        # 地理区域特征
        df['hemisphere_north'] = (df['latitude'] > 0).astype(int)
        df['hemisphere_east'] = (df['longitude'] > 0).astype(int)

        # 主要航线区域
        df['pacific_region'] = ((df['longitude'] > 120) | (df['longitude'] < -120)).astype(int)
        df['atlantic_region'] = ((df['longitude'] > -80) & (df['longitude'] < 20)).astype(int)
        df['indian_region'] = ((df['longitude'] > 40) & (df['longitude'] < 120)).astype(int)

    return df


def haversine_distance(lat1, lon1, lat2, lon2):
    """
    计算两点间的球面距离（公里）
    """
    R = 6371  # 地球半径（公里）

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加交互特征
    """
    # 选择主要的数值特征进行交互
    numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    important_cols = []

    # 选择重要特征（基于列名关键词）
    keywords = ['value', 'price', 'congestion', 'weather', 'exchange', 'distance']
    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in keywords):
            important_cols.append(col)

    # 限制交互特征数量，避免特征爆炸
    important_cols = important_cols[:5]

    # 创建两两交互特征
    for i, col1 in enumerate(important_cols):
        for j, col2 in enumerate(important_cols[i+1:], i+1):
            if col1 != col2:
                # 乘积交互
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]

                # 比值交互（避免除零）
                df[f'{col1}_ratio_{col2}'] = df[col1] / (df[col2] + 1e-8)

    logger.info(f"创建了交互特征，基于 {len(important_cols)} 个主要特征")

    return df


def add_statistical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    添加统计特征（滑动窗口统计）
    """
    time_column = 'timestamp'
    if time_column in df.columns:
        df = df.sort_values(time_column)

        # 选择数值列进行统计
        numeric_cols = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        stat_cols = [col for col in numeric_cols if 'target' not in col.lower()][:3]  # 限制列数

        windows = [7, 30]  # 7天和30天窗口

        for col in stat_cols:
            for window in windows:
                # 滑动平均
                df[f'{col}_ma_{window}'] = df[col].rolling(window=window, min_periods=1).mean()

                # 滑动标准差
                df[f'{col}_std_{window}'] = df[col].rolling(window=window, min_periods=1).std()

                # 与滑动平均的偏离
                df[f'{col}_dev_{window}'] = df[col] - df[f'{col}_ma_{window}']

    return df


def prepare_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    将DataFrame转换为特征矩阵
    """
    # 选择数值型特征
    feature_columns = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns

    # 排除目标变量和ID列
    exclude_columns = ['target', 'id', 'index']
    feature_columns = [col for col in feature_columns if not any(exc in col.lower() for exc in exclude_columns)]

    # 处理缺失值
    feature_df = df[feature_columns].copy()
    feature_df = feature_df.fillna(feature_df.mean())

    # 处理无穷值
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
    feature_df = feature_df.fillna(0)

    logger.info(f"准备特征矩阵: {len(feature_columns)} 个特征")

    return feature_df.values


if __name__ == "__main__":
    # 测试代码
    import yaml
    from data_loader import load_raw_data
    from preprocessor import preprocess

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载和预处理数据
    raw_data = load_raw_data(config['data'])
    train_df, val_df, test_df = preprocess(raw_data, config['data'])

    # 特征工程
    train_features = extract_features(train_df, config['features'])
    val_features = extract_features(val_df, config['features'])
    test_features = extract_features(test_df, config['features'])

    print(f"\n特征工程结果:")
    print(f"训练集特征形状: {train_features.shape}")
    print(f"验证集特征形状: {val_features.shape}")
    print(f"测试集特征形状: {test_features.shape}")
