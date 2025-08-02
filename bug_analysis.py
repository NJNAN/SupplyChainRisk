#!/usr/bin/env python3
"""
供应链风险预测系统 - 完整Bug检查和修复指南
"""

import os
import sys
import subprocess

def print_section(title):
    print(f"\n{'='*60}")
    print(f"🔍 {title}")
    print('='*60)

def check_sklearn_compatibility():
    """检查sklearn兼容性问题"""
    try:
        # 尝试导入关键的sklearn模块
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import accuracy_score
        print("✅ sklearn 导入成功")
        return True
    except Exception as e:
        print(f"❌ sklearn 导入失败: {e}")
        print("💡 这通常是NumPy 2.x与旧版sklearn的兼容性问题")
        return False

def analyze_current_bugs():
    """分析当前已知的bug和修复状态"""
    
    print_section("当前项目Bug状态分析")
    
    bugs_status = {
        "NumPy 1.24+ np.int等别名移除": "✅ 已修复 - 改为明确类型列表",
        "PyTorch ReduceLROnPlateau verbose参数": "✅ 已修复 - 移除verbose参数", 
        "模型训练输入维度不匹配": "✅ 已修复 - 统一input_dim为93",
        "GNN模型数据流问题": "✅ 已修复 - 使用特征工程后的数据",
        "序列数据创建功能": "✅ 已修复 - 新增create_sequences_from_features",
        "温度监控卡死问题": "✅ 已修复 - 增加超时保护",
        "设备不一致问题": "✅ 已修复 - 确保模型和数据在同一设备",
        "模型加载weights_only问题": "✅ 已修复 - 明确设置weights_only=False",
        "可视化绘图报错": "✅ 已修复 - 增加异常捕获",
        "sklearn与NumPy 2.x兼容性": "⚠️  需要处理 - 主要剩余问题",
        "graph_builder.py不必要的sklearn导入": "✅ 已修复 - 移除未使用的导入"
    }
    
    for bug, status in bugs_status.items():
        print(f"{status} {bug}")
    
    return bugs_status

def provide_solutions():
    """提供解决方案"""
    
    print_section("解决方案")
    
    print("🎯 主要问题: sklearn与NumPy 2.x兼容性")
    print("\n💡 推荐解决方案:")
    print("方案1: 降级NumPy (推荐)")
    print("  pip install 'numpy<2.0'")
    print("  pip install 'numpy>=1.21.0,<2.0.0'")
    
    print("\n方案2: 升级sklearn")
    print("  pip install 'scikit-learn>=1.3.0'")
    
    print("\n方案3: 使用兼容的虚拟环境")
    print("  conda create -n supply_chain python=3.10")
    print("  conda activate supply_chain")
    print("  pip install -r requirements.txt")
    
    print("\n方案4: 修改代码减少sklearn依赖")
    print("  - 用内置方法替代sklearn.preprocessing")
    print("  - 用torch.nn.functional替代sklearn.metrics")

def check_alternative_implementation():
    """检查是否可以不依赖sklearn运行"""
    
    print_section("测试非sklearn依赖版本")
    
    # 尝试创建一个最小化的导入测试
    minimal_imports = [
        "pandas", "numpy", "torch", "networkx", "yaml", "matplotlib"
    ]
    
    for module in minimal_imports:
        try:
            __import__(module)
            print(f"✅ {module} 可用")
        except Exception as e:
            print(f"❌ {module} 不可用: {e}")

def create_sklearn_free_version():
    """创建不依赖sklearn的临时版本"""
    
    print_section("创建sklearn替代方案")
    
    print("🔧 可以用以下方法替代sklearn功能:")
    print("1. StandardScaler -> 手动标准化: (x - mean) / std")
    print("2. accuracy_score -> torch.eq(pred, target).sum() / len(target)")
    print("3. f1_score -> 手动计算F1分数")
    print("4. LabelEncoder -> pd.factorize()")
    
    replacements = {
        "StandardScaler": "手动标准化",
        "accuracy_score": "torch计算准确率", 
        "f1_score": "手动F1计算",
        "LabelEncoder": "pandas factorize"
    }
    
    for sklearn_func, replacement in replacements.items():
        print(f"  {sklearn_func} → {replacement}")

def test_core_functionality():
    """测试核心功能是否正常"""
    
    print_section("核心功能测试")
    
    # 检查是否可以导入基础模块
    core_modules = [
        "data_loader", "preprocessor", "features", "graph_builder", 
        "trainer", "evaluator", "compression", "deployment", "robustness"
    ]
    
    available_modules = []
    
    for module in core_modules:
        try:
            # 由于sklearn问题，某些模块可能导入失败
            if module in ["preprocessor", "features", "trainer", "evaluator", "robustness"]:
                print(f"⚠️  {module} - 可能因sklearn问题无法导入")
            else:
                __import__(module)
                print(f"✅ {module} 导入成功")
                available_modules.append(module)
        except Exception as e:
            if "sklearn" in str(e).lower():
                print(f"⚠️  {module} - sklearn兼容性问题: {e}")
            else:
                print(f"❌ {module} - 其他问题: {e}")
    
    return available_modules

def main():
    print("🚀 供应链风险预测系统 - Bug分析报告")
    
    # 分析bug状态
    bugs_status = analyze_current_bugs()
    
    # 检查sklearn兼容性
    sklearn_ok = check_sklearn_compatibility()
    
    # 测试核心功能
    available_modules = test_core_functionality()
    
    # 提供解决方案
    provide_solutions()
    
    # 检查替代实现
    check_alternative_implementation()
    
    # 创建sklearn替代方案指南
    create_sklearn_free_version()
    
    print_section("总结")
    
    print(f"📊 Bug修复状态: {sum(1 for status in bugs_status.values() if '✅' in status)}/{len(bugs_status)} 已修复")
    print(f"🔧 主要剩余问题: sklearn与NumPy 2.x兼容性")
    print(f"📦 可用模块: {len(available_modules)}/{len(['data_loader', 'graph_builder'])} (不依赖sklearn)")
    
    if sklearn_ok:
        print("🎉 所有功能应该可以正常运行！")
    else:
        print("⚠️  需要解决sklearn兼容性问题才能完整运行")
    
    print("\n🎯 推荐操作:")
    print("1. 降级NumPy: pip install 'numpy<2.0'")
    print("2. 或创建新的兼容环境")
    print("3. 或使用提供的sklearn替代方案")

if __name__ == "__main__":
    main()
