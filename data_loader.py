"""
数据加载模块 - 从UN Comtrade API和AIS数据源加载贸易和航运数据
"""

import pandas as pd
import numpy as np
import requests
import time
import json
from typing import Dict, Optional
from tqdm import tqdm
import logging
import os

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(config: Dict) -> Dict[str, pd.DataFrame]:
    """
    根据配置加载原始数据

    Args:
        config: 配置字典，包含数据源配置信息

    Returns:
        包含'comtrade', 'ais', 'synthetic'三个DataFrame的字典
    """
    data_dict = {}

    try:
        # 加载UN Comtrade数据
        logger.info("开始加载UN Comtrade数据...")
        data_dict['comtrade'] = load_comtrade_data(config)

        # 加载AIS数据
        logger.info("开始加载AIS数据...")
        data_dict['ais'] = load_ais_data(config)

        # 生成合成数据（如果配置启用）
        if config.get('use_synthetic', False):
            logger.info("开始生成合成数据...")
            data_dict['synthetic'] = generate_synthetic_data(config)
        else:
            data_dict['synthetic'] = pd.DataFrame()

        logger.info("数据加载完成！")
        return data_dict

    except Exception as e:
        logger.error(f"数据加载失败: {str(e)}")
        raise


def load_comtrade_data(config: Dict) -> pd.DataFrame:
    """
    从UN Comtrade API加载贸易数据
    """
    try:
        # 如果本地存在缓存文件，直接加载
        cache_file = "data/comtrade_cache.csv"
        if os.path.exists(cache_file):
            logger.info("从缓存文件加载Comtrade数据...")
            return pd.read_csv(cache_file)

        # 否则从API获取数据
        api_url = config.get('comtrade_api', 'https://comtrade.un.org/api/get')

        # 构建API请求参数
        params = {
            'max': 50000,  # 最大记录数
            'type': 'C',   # 商品贸易
            'freq': 'M',   # 月度数据
            'px': 'HS',    # HS分类
            'ps': '2023',  # 年份
            'r': 'all',    # 所有报告国
            'p': 'all',    # 所有贸易伙伴
            'rg': 'all',   # 进出口
            'cc': 'AG2',   # 2位HS代码
            'fmt': 'json'  # JSON格式
        }

        logger.info("正在从UN Comtrade API获取数据...")
        response = requests.get(api_url, params=params, timeout=30)
        response.raise_for_status()

        data = response.json()

        if 'dataset' in data:
            df = pd.DataFrame(data['dataset'])

            # 数据清洗和标准化
            df = clean_comtrade_data(df)

            # 保存缓存
            os.makedirs("data", exist_ok=True)
            df.to_csv(cache_file, index=False)

            return df
        else:
            raise ValueError("API返回数据格式异常")

    except requests.exceptions.RequestException as e:
        logger.warning(f"API请求失败: {str(e)}，使用模拟数据")
        return generate_mock_comtrade_data(config)
    except Exception as e:
        logger.warning(f"Comtrade数据加载失败: {str(e)}，使用模拟数据")
        return generate_mock_comtrade_data(config)


def clean_comtrade_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗和标准化Comtrade数据
    """
    # 重命名关键列
    column_mapping = {
        'yr': 'year',
        'period': 'month',
        'rtTitle': 'reporter',
        'ptTitle': 'partner',
        'cmdDescE': 'commodity',
        'TradeValue': 'trade_value',
        'NetWeight': 'net_weight',
        'rgDesc': 'trade_flow'
    }

    for old_name, new_name in column_mapping.items():
        if old_name in df.columns:
            df[new_name] = df[old_name]

    # 创建时间戳
    df['timestamp'] = pd.to_datetime(
        df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'
    )

    # 处理缺失值
    df['trade_value'] = pd.to_numeric(df['trade_value'], errors='coerce').fillna(0)
    df['net_weight'] = pd.to_numeric(df['net_weight'], errors='coerce').fillna(0)

    # 添加目标变量（货物延误标志）
    # 基于贸易价值异常波动来模拟延误
    df['delay_flag'] = (df['trade_value'] < df['trade_value'].quantile(0.1)).astype(int)

    return df


def load_ais_data(config: Dict) -> pd.DataFrame:
    """
    加载AIS船舶轨迹数据
    """
    try:
        ais_source = config.get('ais_source', 'local')

        if ais_source == 'local':
            # 从本地文件加载
            ais_file = "data/ais_data.csv"
            if os.path.exists(ais_file):
                logger.info("从本地文件加载AIS数据...")
                df = pd.read_csv(ais_file)
            else:
                logger.warning("本地AIS文件不存在，生成模拟数据")
                df = generate_mock_ais_data(config)
        else:
            # 从在线源获取（这里用模拟数据代替）
            logger.info("生成模拟AIS数据...")
            df = generate_mock_ais_data(config)

        return clean_ais_data(df)

    except Exception as e:
        logger.warning(f"AIS数据加载失败: {str(e)}，使用模拟数据")
        return generate_mock_ais_data(config)


def clean_ais_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗AIS数据
    """
    # 确保时间列存在
    if 'timestamp' not in df.columns:
        df['timestamp'] = pd.date_range(start='2023-01-01', periods=len(df), freq='H')

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 处理坐标数据
    df['latitude'] = pd.to_numeric(df.get('latitude', 0), errors='coerce')
    df['longitude'] = pd.to_numeric(df.get('longitude', 0), errors='coerce')

    # 计算速度（如果不存在）
    if 'speed' not in df.columns:
        df['speed'] = np.random.normal(10, 3, len(df))  # 平均10节，标准差3

    # 添加异常订单标志（基于速度异常）
    df['anomaly_flag'] = ((df['speed'] < 2) | (df['speed'] > 25)).astype(int)

    return df


def generate_mock_comtrade_data(config: Dict) -> pd.DataFrame:
    """
    生成模拟Comtrade数据
    """
    n_samples = config.get('synthetic_samples', 10000) // 2

    # 港口列表
    ports = ['Shanghai', 'Singapore', 'Rotterdam', 'Hamburg', 'Los Angeles',
             'Long Beach', 'Hong Kong', 'Busan', 'Ningbo', 'Guangzhou']

    # HS代码列表
    hs_codes = ['01', '02', '03', '04', '05', '84', '85', '87', '90', '99']

    # 生成数据
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
        'reporter': np.random.choice(ports, n_samples),
        'partner': np.random.choice(ports, n_samples),
        'hs_code': np.random.choice(hs_codes, n_samples),
        'trade_value': np.random.lognormal(10, 1, n_samples),
        'net_weight': np.random.lognormal(8, 1, n_samples),
        'trade_flow': np.random.choice(['Import', 'Export'], n_samples)
    }

    df = pd.DataFrame(data)

    # 添加延误标志
    df['delay_flag'] = np.random.binomial(1, 0.15, n_samples)  # 15%延误率

    return df


def generate_mock_ais_data(config: Dict) -> pd.DataFrame:
    """
    生成模拟AIS数据
    """
    n_samples = config.get('synthetic_samples', 10000) // 2

    # 主要航线区域
    regions = [
        {'name': 'Pacific', 'lat_range': (20, 50), 'lon_range': (120, 180)},
        {'name': 'Atlantic', 'lat_range': (10, 60), 'lon_range': (-80, 20)},
        {'name': 'Indian', 'lat_range': (-20, 30), 'lon_range': (40, 120)}
    ]

    data = []
    for i in range(n_samples):
        region = np.random.choice(regions)

        record = {
            'timestamp': pd.Timestamp('2023-01-01') + pd.Timedelta(hours=i),
            'vessel_id': f'V{i//100:04d}',  # 每100条记录一艘船
            'latitude': np.random.uniform(*region['lat_range']),
            'longitude': np.random.uniform(*region['lon_range']),
            'speed': max(0, np.random.normal(12, 4)),  # 平均12节
            'course': np.random.uniform(0, 360),
            'vessel_type': np.random.choice(['Container', 'Bulk', 'Tanker', 'General'])
        }
        data.append(record)

    df = pd.DataFrame(data)

    # 添加异常订单标志
    df['anomaly_flag'] = ((df['speed'] < 3) | (df['speed'] > 20)).astype(int)

    return df


def generate_synthetic_data(config: Dict) -> pd.DataFrame:
    """
    生成完全合成的供应链数据
    """
    n_samples = config.get('synthetic_samples', 10000)

    # 生成多维特征
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=n_samples, freq='H'),
        'port_origin': np.random.choice(['Port_A', 'Port_B', 'Port_C', 'Port_D'], n_samples),
        'port_destination': np.random.choice(['Port_W', 'Port_X', 'Port_Y', 'Port_Z'], n_samples),
        'cargo_type': np.random.choice(['Electronics', 'Textiles', 'Machinery', 'Food'], n_samples),
        'cargo_value': np.random.lognormal(12, 1.5, n_samples),
        'weather_score': np.random.normal(0.7, 0.2, n_samples),
        'congestion_index': np.random.exponential(0.3, n_samples),
        'holiday_flag': np.random.binomial(1, 0.1, n_samples)
    }

    df = pd.DataFrame(data)

    # 基于多个因素生成风险标签
    risk_score = (
        0.3 * (df['weather_score'] < 0.5).astype(int) +
        0.4 * (df['congestion_index'] > 0.5).astype(int) +
        0.2 * df['holiday_flag'] +
        0.1 * np.random.random(n_samples)
    )

    df['risk_flag'] = (risk_score > 0.3).astype(int)

    return df


if __name__ == "__main__":
    # 测试代码
    import yaml

    with open('config.yaml', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 加载数据
    raw_data = load_raw_data(config['data'])

    # 打印数据统计信息
    for data_type, df in raw_data.items():
        if not df.empty:
            print(f"\n{data_type.upper()} 数据统计:")
            print(f"  形状: {df.shape}")
            print(f"  列名: {list(df.columns)}")
            print(f"  时间范围: {df['timestamp'].min()} - {df['timestamp'].max()}")
