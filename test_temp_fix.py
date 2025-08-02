#!/usr/bin/env python3
"""
测试温度监控修复
"""
import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO)

# 创建一个简单的测试模型
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.linear(x)

# 测试温度监控
from deployment import DeploymentBenchmark

def test_temperature_monitoring():
    print("🔧 测试温度监控修复...")
    
    config = {'deployment': {}}
    benchmark = DeploymentBenchmark(config)
    
    model = SimpleModel()
    sample_input = torch.randn(1, 10)
    
    # 测试温度监控（缩短时间到3秒）
    result = benchmark.temperature_monitoring(model, sample_input, duration_seconds=3)
    
    print(f"✅ 温度监控测试完成")
    print(f"结果: {result}")
    
    return result

if __name__ == "__main__":
    test_temperature_monitoring()
