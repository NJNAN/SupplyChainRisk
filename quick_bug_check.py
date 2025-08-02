#!/usr/bin/env python3
"""
快速bug检查脚本 - 不依赖sklearn
"""

import pandas as pd
import numpy as np
import yaml
import os
import sys

def test_basic_modules():
    """测试基础模块导入"""
    print("🔍 测试基础模块导入...")
    
    try:
        import data_loader
        print("✅ data_loader 模块导入成功")
    except Exception as e:
        print(f"❌ data_loader 导入失败: {e}")
        return False
    
    try:
        # 测试配置文件读取
        with open('config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ config.yaml 读取成功")
    except Exception as e:
        print(f"❌ config.yaml 读取失败: {e}")
        return False
    
    try:
        # 测试数据加载
        raw_data = data_loader.load_raw_data(config['data'])
        print(f"✅ 数据加载成功，数据源数量: {len(raw_data)}")
        for source, df in raw_data.items():
            print(f"  - {source}: {df.shape}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return False
    
    return True

def test_numpy_compatibility():
    """测试NumPy兼容性问题"""
    print("\n🔍 测试NumPy兼容性...")
    
    try:
        # 测试select_dtypes的新语法
        df = pd.DataFrame({
            'int_col': [1, 2, 3],
            'float_col': [1.1, 2.2, 3.3],
            'str_col': ['a', 'b', 'c']
        })
        
        # 旧语法（可能有问题）
        try:
            numeric_cols_old = df.select_dtypes(include=[np.number]).columns
            print("✅ select_dtypes(include=[np.number]) 工作正常")
        except Exception as e:
            print(f"⚠️  select_dtypes(include=[np.number]) 失败: {e}")
        
        # 新语法（应该工作）
        numeric_cols_new = df.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
        print("✅ select_dtypes 明确类型列表工作正常")
        
        return True
    except Exception as e:
        print(f"❌ NumPy兼容性测试失败: {e}")
        return False

def test_graph_builder():
    """测试图构建模块"""
    print("\n🔍 测试图构建模块...")
    
    try:
        # 简单测试，不使用sklearn.metrics.pairwise
        import networkx as nx
        import torch
        print("✅ networkx 和 torch 导入成功")
        
        # 测试基本的图创建
        G = nx.Graph()
        G.add_node(0, lat=0.0, lon=0.0)
        G.add_node(1, lat=1.0, lon=1.0)
        G.add_edge(0, 1, weight=1.0)
        print("✅ NetworkX 基本图操作成功")
        
        # 测试torch geometric
        try:
            from torch_geometric.data import Data
            x = torch.FloatTensor([[1, 2], [3, 4]])
            edge_index = torch.LongTensor([[0, 1], [1, 0]])
            data = Data(x=x, edge_index=edge_index)
            print("✅ PyTorch Geometric 数据创建成功")
        except Exception as e:
            print(f"⚠️  PyTorch Geometric 可能有问题: {e}")
        
        return True
    except Exception as e:
        print(f"❌ 图构建模块测试失败: {e}")
        return False

def test_model_imports():
    """测试模型导入"""
    print("\n🔍 测试模型导入...")
    
    try:
        from src.models import gnn_models, rnn_models, transformer
        print("✅ 所有模型模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 模型导入失败: {e}")
        return False

def main():
    print("🚀 供应链风险预测系统 - Bug检查")
    print("=" * 50)
    
    all_passed = True
    
    # 基础模块测试
    if not test_basic_modules():
        all_passed = False
    
    # NumPy兼容性测试
    if not test_numpy_compatibility():
        all_passed = False
    
    # 图构建测试
    if not test_graph_builder():
        all_passed = False
    
    # 模型导入测试
    if not test_model_imports():
        all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 所有基础测试通过！主要问题可能是sklearn与NumPy 2.x的兼容性")
        print("💡 建议:")
        print("  1. 降级NumPy到1.x版本: pip install 'numpy<2.0'")
        print("  2. 或升级scikit-learn到兼容NumPy 2.x的版本")
    else:
        print("❌ 发现一些问题需要修复")

if __name__ == "__main__":
    main()
