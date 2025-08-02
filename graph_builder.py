"""
图结构构造模块 - 为GNN模型构建图数据结构
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
import networkx as nx
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import from_networkx
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class GraphData:
    """图数据类，兼容PyTorch Geometric格式"""

    def __init__(self, x, edge_index, edge_attr=None, y=None):
        self.x = x  # 节点特征 [num_nodes, num_node_features]
        self.edge_index = edge_index  # 边索引 [2, num_edges]
        self.edge_attr = edge_attr  # 边特征 [num_edges, num_edge_features]
        self.y = y  # 图标签或节点标签

    def to_torch_geometric(self):
        """转换为PyTorch Geometric Data对象"""
        return Data(
            x=torch.FloatTensor(self.x),
            edge_index=torch.LongTensor(self.edge_index),
            edge_attr=torch.FloatTensor(self.edge_attr) if self.edge_attr is not None else None,
            y=torch.LongTensor([self.y]) if isinstance(self.y, (int, float)) else torch.FloatTensor(self.y)
        )


def build_graphs(df: pd.DataFrame, config: Dict) -> List[GraphData]:
    """
    构建图数据列表

    Args:
        df: 输入数据框
        config: 配置参数

    Returns:
        GraphData对象列表
    """
    logger.info("开始构建图结构...")

    graphs = []
    as_sequence = config.get('as_sequence', False)

    if as_sequence:
        # 同时生成图和序列数据
        graphs, sequences = build_temporal_graphs(df, config)
        return graphs, sequences
    else:
        # 只生成图数据
        if 'order_id' in df.columns or 'shipment_id' in df.columns:
            # 按订单/货运构建图
            graphs = build_shipment_graphs(df, config)
        else:
            # 构建全局供应链图
            graphs = build_supply_chain_graph(df, config)

    logger.info(f"构建了 {len(graphs)} 个图")
    return graphs


def build_shipment_graphs(df: pd.DataFrame, config: Dict) -> List[GraphData]:
    """
    为每个订单/货运构建独立的图
    """
    graphs = []

    # 确定分组列
    group_col = 'order_id' if 'order_id' in df.columns else 'shipment_id'
    if group_col not in df.columns:
        # 创建虚拟订单ID
        df['order_id'] = df.index // 10  # 每10行数据作为一个订单
        group_col = 'order_id'

    for order_id, order_data in df.groupby(group_col):
        if len(order_data) < 2:  # 至少需要2个节点
            continue

        graph = build_single_shipment_graph(order_data, config)
        if graph is not None:
            graphs.append(graph)

    return graphs


def build_single_shipment_graph(order_data: pd.DataFrame, config: Dict) -> Optional[GraphData]:
    """
    为单个订单构建图
    """
    try:
        # 提取港口/地点作为节点
        nodes = extract_nodes_from_shipment(order_data)
        if len(nodes) < 2:
            return None

        # 构建节点特征矩阵
        node_features = create_node_features(nodes, order_data, config)

        # 构建边和边特征
        edge_index, edge_features = create_edges_from_route(nodes, order_data, config)

        # 图级别标签（基于整个订单的风险）
        graph_label = determine_graph_label(order_data)

        return GraphData(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            y=graph_label
        )

    except Exception as e:
        logger.warning(f"构建订单图失败: {str(e)}")
        return None


def extract_nodes_from_shipment(order_data: pd.DataFrame) -> List[Dict]:
    """
    从订单数据中提取节点（港口/清关点）
    """
    nodes = []

    # 尝试从不同列中提取地点信息
    location_columns = []
    for col in order_data.columns:
        if any(keyword in col.lower() for keyword in ['port', 'location', 'city', 'country']):
            location_columns.append(col)

    if not location_columns:
        # 如果没有地点列，基于数据行创建虚拟节点
        for idx, row in order_data.iterrows():
            nodes.append({
                'id': f'node_{len(nodes)}',
                'type': 'waypoint',
                'timestamp': row.get('timestamp', pd.Timestamp.now()),
                'latitude': row.get('latitude', np.random.uniform(-90, 90)),
                'longitude': row.get('longitude', np.random.uniform(-180, 180)),
                'data': row.to_dict()
            })
    else:
        # 基于地点信息创建节点
        unique_locations = set()
        for _, row in order_data.iterrows():
            location = str(row[location_columns[0]]) if location_columns else f"loc_{len(nodes)}"
            if location not in unique_locations:
                unique_locations.add(location)
                nodes.append({
                    'id': location,
                    'type': 'port' if 'port' in location.lower() else 'location',
                    'timestamp': row.get('timestamp', pd.Timestamp.now()),
                    'latitude': row.get('latitude', np.random.uniform(-90, 90)),
                    'longitude': row.get('longitude', np.random.uniform(-180, 180)),
                    'data': row.to_dict()
                })

    return nodes


def create_node_features(nodes: List[Dict], order_data: pd.DataFrame, config: Dict) -> np.ndarray:
    """
    创建节点特征矩阵
    """
    feature_dim = config.get('node_feature_dim', 32)
    node_features = []

    for node in nodes:
        features = []

        # 地理特征
        features.extend([
            node['latitude'] / 90.0,  # 标准化纬度
            node['longitude'] / 180.0,  # 标准化经度
        ])

        # 时间特征
        timestamp = pd.to_datetime(node['timestamp'])
        features.extend([
            timestamp.hour / 24.0,
            timestamp.dayofweek / 7.0,
            timestamp.month / 12.0
        ])

        # 节点类型编码
        node_type_encoding = {
            'port': [1, 0, 0],
            'location': [0, 1, 0],
            'waypoint': [0, 0, 1]
        }
        features.extend(node_type_encoding.get(node['type'], [0, 0, 1]))

        # 从订单数据中提取其他特征
        node_data = node['data']
        for key in ['trade_value', 'congestion_index', 'weather_score']:
            if key in node_data:
                features.append(float(node_data[key]))
            else:
                features.append(0.0)

        # 填充或截断到固定维度
        if len(features) < feature_dim:
            features.extend([0.0] * (feature_dim - len(features)))
        else:
            features = features[:feature_dim]

        node_features.append(features)

    return np.array(node_features, dtype=np.float32)


def create_edges_from_route(nodes: List[Dict], order_data: pd.DataFrame, config: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    从路线创建边和边特征
    """
    num_nodes = len(nodes)
    edges = []
    edge_features = []

    # 按时间排序节点
    sorted_nodes = sorted(enumerate(nodes), key=lambda x: x[1]['timestamp'])

    # 创建顺序连接的边（路线）
    for i in range(len(sorted_nodes) - 1):
        src_idx, src_node = sorted_nodes[i]
        dst_idx, dst_node = sorted_nodes[i + 1]

        # 添加有向边
        edges.append([src_idx, dst_idx])

        # 计算边特征
        edge_feature = calculate_edge_features(src_node, dst_node, order_data)
        edge_features.append(edge_feature)

        # 可选：添加反向边
        if config.get('bidirectional_edges', True):
            edges.append([dst_idx, src_idx])
            edge_features.append(edge_feature)  # 使用相同的边特征

    # 添加基于距离的额外连接
    if config.get('distance_based_edges', True):
        threshold_km = config.get('distance_threshold', 1000)  # 1000公里

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                distance = haversine_distance_single(
                    nodes[i]['latitude'], nodes[i]['longitude'],
                    nodes[j]['latitude'], nodes[j]['longitude']
                )

                if distance < threshold_km:
                    # 添加双向边
                    edges.extend([[i, j], [j, i]])

                    edge_feature = [
                        distance / 1000.0,  # 标准化距离
                        0.0,  # 时间差（非时序边）
                        0.5   # 连接强度
                    ]
                    edge_features.extend([edge_feature, edge_feature])

    if not edges:
        # 如果没有边，创建一个自环
        edges = [[0, 0]]
        edge_features = [[0.0, 0.0, 0.0]]

    edge_index = np.array(edges).T  # 转置为 [2, num_edges]
    edge_attr = np.array(edge_features, dtype=np.float32)

    return edge_index, edge_attr


def calculate_edge_features(src_node: Dict, dst_node: Dict, order_data: pd.DataFrame) -> List[float]:
    """
    计算边特征
    """
    # 地理距离
    distance = haversine_distance_single(
        src_node['latitude'], src_node['longitude'],
        dst_node['latitude'], dst_node['longitude']
    )

    # 时间差
    time_diff = (pd.to_datetime(dst_node['timestamp']) - pd.to_datetime(src_node['timestamp'])).total_seconds() / 3600  # 小时

    # 运输成本估算（基于距离和时间）
    transport_cost = distance * 0.001 + abs(time_diff) * 0.01

    return [
        distance / 1000.0,  # 标准化距离 (km -> 1000km单位)
        min(time_diff / 24.0, 10.0),  # 标准化时间差 (小时 -> 天，上限10天)
        min(transport_cost, 10.0)  # 标准化运输成本
    ]


def haversine_distance_single(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    计算两点间距离（公里）
    """
    R = 6371  # 地球半径

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def determine_graph_label(order_data: pd.DataFrame) -> int:
    """
    确定图级别的标签
    """
    # 基于目标列或多个风险指标
    if 'target' in order_data.columns:
        return int(order_data['target'].max())  # 订单中任何节点有风险即为高风险
    elif 'delay_flag' in order_data.columns:
        return int(order_data['delay_flag'].max())
    elif 'anomaly_flag' in order_data.columns:
        return int(order_data['anomaly_flag'].max())
    else:
        # 基于其他指标判断风险
        risk_indicators = []

        if 'congestion_index' in order_data.columns:
            risk_indicators.append(order_data['congestion_index'].mean() > 0.5)
        if 'weather_score' in order_data.columns:
            risk_indicators.append(order_data['weather_score'].mean() < 0.3)

        return int(any(risk_indicators)) if risk_indicators else 0


def build_supply_chain_graph(df: pd.DataFrame, config: Dict) -> List[GraphData]:
    """
    构建全局供应链图
    """
    # 提取所有唯一地点作为节点
    locations = extract_all_locations(df)

    # 构建全局图
    G = nx.Graph()

    # 添加节点
    for i, location in enumerate(locations):
        G.add_node(i, **location)

    # 基于贸易流添加边
    add_trade_edges(G, df, locations)

    # 转换为GraphData格式
    graph_data = networkx_to_graph_data(G, df)

    return [graph_data]


def extract_all_locations(df: pd.DataFrame) -> List[Dict]:
    """
    提取所有唯一地点
    """
    locations = []
    seen_locations = set()

    location_columns = [col for col in df.columns if any(keyword in col.lower() for keyword in ['port', 'location', 'reporter', 'partner'])]

    for _, row in df.iterrows():
        for col in location_columns:
            if col in row and pd.notna(row[col]):
                location_name = str(row[col])
                if location_name not in seen_locations:
                    seen_locations.add(location_name)
                    locations.append({
                        'name': location_name,
                        'latitude': row.get('latitude', np.random.uniform(-60, 60)),
                        'longitude': row.get('longitude', np.random.uniform(-180, 180)),
                        'type': 'port' if 'port' in location_name.lower() else 'location'
                    })

    return locations


def add_trade_edges(G: nx.Graph, df: pd.DataFrame, locations: List[Dict]):
    """
    基于贸易数据添加边
    """
    location_to_idx = {loc['name']: i for i, loc in enumerate(locations)}

    # 基于贸易关系添加边
    for _, row in df.iterrows():
        src_name = None
        dst_name = None

        # 寻找源和目标地点
        if 'reporter' in row and 'partner' in row:
            src_name = str(row['reporter'])
            dst_name = str(row['partner'])
        elif 'port_origin' in row and 'port_destination' in row:
            src_name = str(row['port_origin'])
            dst_name = str(row['port_destination'])

        if src_name and dst_name and src_name in location_to_idx and dst_name in location_to_idx:
            src_idx = location_to_idx[src_name]
            dst_idx = location_to_idx[dst_name]

            # 累计贸易量作为边权重
            if G.has_edge(src_idx, dst_idx):
                G[src_idx][dst_idx]['weight'] += row.get('trade_value', 1)
            else:
                G.add_edge(src_idx, dst_idx, weight=row.get('trade_value', 1))


def networkx_to_graph_data(G: nx.Graph, df: pd.DataFrame) -> GraphData:
    """
    将NetworkX图转换为GraphData
    """
    # 节点特征
    node_features = []
    for node_id in G.nodes():
        node_data = G.nodes[node_id]
        features = [
            node_data.get('latitude', 0) / 90.0,
            node_data.get('longitude', 0) / 180.0,
            1.0 if node_data.get('type') == 'port' else 0.0
        ]
        node_features.append(features)

    # 边索引和边特征
    edge_index = []
    edge_features = []

    for src, dst, edge_data in G.edges(data=True):
        edge_index.extend([[src, dst], [dst, src]])  # 双向边
        weight = edge_data.get('weight', 1.0)
        edge_feature = [np.log1p(weight)]  # log变换的权重
        edge_features.extend([edge_feature, edge_feature])

    if not edge_index:
        # 如果没有边，添加自环
        edge_index = [[0, 0]]
        edge_features = [[0.0]]

    # 图级别标签（基于整体风险）
    graph_label = int(df.get('target', 0).mean() > 0.5) if 'target' in df.columns else 0

    return GraphData(
        x=np.array(node_features, dtype=np.float32),
        edge_index=np.array(edge_index).T,
        edge_attr=np.array(edge_features, dtype=np.float32),
        y=graph_label
    )


def build_temporal_graphs(df: pd.DataFrame, config: Dict) -> Tuple[List[GraphData], List[np.ndarray]]:
    """
    构建时序图和对应的序列数据
    """
    graphs = []
    sequences = []

    sequence_length = config.get('sequence_length', 24)

    # 按时间排序
    df = df.sort_values('timestamp')

    # 滑动窗口创建时序图
    for i in range(len(df) - sequence_length + 1):
        window_data = df.iloc[i:i + sequence_length]

        # 构建图
        graph = build_single_shipment_graph(window_data, config)
        if graph is not None:
            graphs.append(graph)

            # 对应的序列数据
            sequence = window_data.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).values
            sequences.append(sequence)

    return graphs, sequences


def create_graph_dataloader(graphs: List[GraphData], batch_size: int = 32, shuffle: bool = True) -> DataLoader:
    """
    创建图数据的DataLoader
    """
    # 转换为PyTorch Geometric Data对象
    torch_graphs = [graph.to_torch_geometric() for graph in graphs]

    return DataLoader(torch_graphs, batch_size=batch_size, shuffle=shuffle)


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

    # 构建图
    train_graphs = build_graphs(train_df, config.get('graph', {}))
    val_graphs = build_graphs(val_df, config.get('graph', {}))
    test_graphs = build_graphs(test_df, config.get('graph', {}))

    print(f"\n图构建结果:")
    print(f"训练图数量: {len(train_graphs)}")
    print(f"验证图数量: {len(val_graphs)}")
    print(f"测试图数量: {len(test_graphs)}")

    if train_graphs:
        sample_graph = train_graphs[0]
        print(f"样本图 - 节点数: {sample_graph.x.shape[0]}, 边数: {sample_graph.edge_index.shape[1]}")
        print(f"节点特征维度: {sample_graph.x.shape[1]}")
        print(f"边特征维度: {sample_graph.edge_attr.shape[1] if sample_graph.edge_attr is not None else 0}")
