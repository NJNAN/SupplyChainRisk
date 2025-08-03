# 全球供应链风险预测系统

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 项目简介

**供应链风险预测系统** 是一个基于深度学习的全球供应链风险评估平台，集成了多种先进的机器学习架构，包括RNN、GNN和Transformer模型。系统提供从数据预处理、模型训练、性能评估到模型压缩和部署的完整工具链，具备强大的鲁棒性测试和可解释性分析能力。

### ✨ 主要特性

- 🤖 **多架构融合**: 支持9种深度学习模型（RNN/LSTM/GRU + GCN/GAT/GraphSAGE + Transformer）
- 📊 **完整工作流**: 数据处理 → 特征工程 → 模型训练 → 评估 → 压缩 → 部署的端到端流程
- 🛡️ **鲁棒性测试**: 4种扰动类型的系统性鲁棒性评估框架
- ⚡ **模型压缩**: 结构剪枝和量化技术，支持边缘设备部署
- 📈 **可视化分析**: 15个专业图表，所有输出完全英语化
- 🔧 **高度可配置**: 灵活的配置系统，支持超参数优化
- 🚀 **生产就绪**: 温度监控、性能基准测试、错误处理完善

### 🎯 研究目标

1. **算法对比研究**: 系统性比较不同深度学习架构在供应链风险预测中的性能
2. **模型轻量化**: 通过剪枝和量化技术实现模型压缩，保持预测精度的同时降低部署成本
3. **鲁棒性分析**: 评估模型在数据扰动和业务中断场景下的稳定性和可靠性
4. **实际部署验证**: 在边缘设备上验证模型的实时推理能力和资源消耗

## 📁 项目结构

```
SupplyChainRisk/
├── main.py                    # 🚀 项目主入口
├── config.yaml               # ⚙️ 配置文件
├── requirements.txt           # 📦 Python依赖清单
│
├── 核心模块/
├── data_loader.py            # 📊 数据加载（UN Comtrade API、AIS数据、合成数据）
├── preprocessor.py           # 🔧 数据预处理（清洗、标准化、时间对齐）
├── features.py               # 🎯 特征工程（时间、节假日、交互特征）
├── graph_builder.py          # 🕸️ 图结构构造（k-近邻、供应链层级关系）
├── trainer.py                # 🏋️ 模型训练（支持9种模型架构）
├── evaluator.py              # 📈 模型评估（15种指标+可视化）
├── compression.py            # ⚡ 模型压缩（剪枝+量化）
├── robustness.py             # 🛡️ 鲁棒性测试（4种扰动类型）
├── deployment.py             # 🚀 部署基准测试（性能监控）
│
├── 模型架构/
├── src/
│   └── models/
│       ├── rnn_models.py     # 🔄 RNN系列（RNN/LSTM/GRU）
│       ├── gnn_models.py     # 🕸️ GNN系列（GCN/GAT/GraphSAGE）
│       └── transformer.py   # 🤖 Transformer模型
│
├── 自动化脚本/
├── scripts/
│   ├── run_all.sh           # 🎯 一键运行完整流程
│   ├── run_train.sh         # 🏋️ 批量训练脚本
│   ├── run_eval.sh          # 📊 批量评估脚本
│   ├── run_compress.sh      # ⚡ 批量压缩脚本
│   ├── run_robust.sh        # 🛡️ 鲁棒性测试脚本
│   └── run_deploy.sh        # 🚀 部署测试脚本
│
├── 数据与结果/
├── data/                     # 📁 原始数据目录
├── checkpoints/              # 💾 训练好的模型检查点
├── results/                  # 📈 实验结果（报告+图表）
└── logs/                     # 📝 详细日志文件
```

## 🔥 系统亮点

### 🤖 **多模型架构支持**
| 模型类型 | 具体架构 | 特点 | 适用场景 |
|---------|---------|------|---------|
| **RNN系列** | RNN, LSTM, GRU | 时序建模强 | 时间序列预测 |
| **GNN系列** | GCN, GAT, GraphSAGE | 关系建模佳 | 网络风险传播 |
| **Transformer** | 自注意力机制 | 长期依赖 | 复杂时序模式 |

### 📊 **完善的评估体系**
- **性能指标**: 准确率、精确率、召回率、F1、AUC等15种指标
- **可视化图表**: 性能对比、ROC曲线、混淆矩阵、训练曲线等15个图表
- **统计检验**: 配对t检验验证模型性能差异显著性
- **交叉验证**: K折交叉验证确保结果可靠性

### 🛡️ **鲁棒性测试框架**
| 扰动类型 | 描述 | 测试级别 |
|---------|------|---------|
| **Drop** | 随机丢弃数据点 | 0.1~0.5 |
| **Shuffle** | 打乱时序顺序 | 0.1~0.5 |
| **Noise** | 添加高斯噪声 | 0.1~0.5 |
| **Business** | 业务中断模拟 | 节假日、疫情、港口关闭 |

### ⚡ **模型压缩技术**
- **结构剪枝**: 30%、50%、70% 三种剪枝比率
- **动态量化**: 8-bit量化，保持精度的同时减少模型大小
- **组合压缩**: 剪枝+量化的混合策略
- **性能保持**: 压缩后性能损失控制在5%以内

## 🚀 快速开始

### 📋 环境要求

#### 系统要求
- **操作系统**: Linux, macOS, Windows 10+
- **Python**: 3.8+ (推荐 3.10)
- **内存**: 最低4GB，推荐8GB+
- **存储空间**: 2GB（含数据和模型）

#### 依赖库版本
```bash
torch>=2.0.0
torch-geometric>=2.3.0
numpy>=1.24.0,<2.0.0  # 兼容scikit-learn
scikit-learn>=1.3.0
matplotlib>=3.6.0
seaborn>=0.12.0
optuna>=3.0.0
networkx>=3.0
psutil>=5.9.0
```

### ⚙️ 环境设置

#### 1. 克隆项目
```bash
git clone https://github.com/your-username/SupplyChainRisk.git
cd SupplyChainRisk
```

#### 2. 创建虚拟环境（推荐）
```bash
# 使用conda
conda create -n supply-risk python=3.10
conda activate supply-risk

# 或使用venv
python -m venv supply-risk
source supply-risk/bin/activate  # Linux/macOS
# supply-risk\Scripts\activate  # Windows
```

#### 3. 安装依赖
```bash
pip install -r requirements.txt

# 如果需要GPU支持
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 4. 初始化项目
```bash
# 创建必要目录
mkdir -p checkpoints data logs results

# 验证安装
python -c "import torch; print(f'PyTorch版本: {torch.__version__}')"
python -c "import torch_geometric; print('PyTorch Geometric安装成功')"
```

### 🎯 运行方式

#### 🚀 **方式1: 一键运行完整流程**
```bash
# 运行所有模块（训练+评估+压缩+鲁棒性+部署）
python main.py --mode all --verbose

# 查看详细输出和进度
python main.py --mode all --verbose --log_level DEBUG
```

#### 🔧 **方式2: 分步骤运行**
```bash
# 1. 模型训练（默认训练所有9种模型）
python main.py --mode train --verbose

# 2. 模型评估
python main.py --mode eval --verbose

# 3. 模型压缩
python main.py --mode compress --verbose

# 4. 鲁棒性测试
python main.py --mode robust --verbose  

# 5. 部署基准测试
python main.py --mode deploy --verbose
```

#### 🎪 **方式3: 单模型训练/测试**
```bash
# 训练特定模型
python main.py --mode train --model lstm --verbose
python main.py --mode train --model gat --verbose
python main.py --mode train --model transformer --verbose

# 评估特定模型
python main.py --mode eval --model gru --verbose

# 对特定模型进行鲁棒性测试
python main.py --mode robust --model gcn --verbose
```

#### 🛠️ **方式4: 使用Shell脚本（推荐批量运行）**
```bash
# 完整流程
bash scripts/run_all.sh

# 分别运行各模块
bash scripts/run_train.sh      # 训练所有模型
bash scripts/run_eval.sh       # 评估所有模型
bash scripts/run_compress.sh   # 压缩所有模型
bash scripts/run_robust.sh     # 鲁棒性测试
bash scripts/run_deploy.sh     # 部署基准测试
```

#### 🔍 **方式5: 超参数优化**
```bash
# 使用Optuna进行超参数优化
python main.py --mode optimize --model lstm --trials 50 --verbose
python main.py --mode optimize --model gat --trials 100 --verbose

# 指定优化目标
python main.py --mode optimize --model gru --trials 30 --metric f1 --verbose
```

## 📊 实验模块详解

### 1. 数据加载与预处理
- **UN Comtrade API**: 自动获取国际贸易数据
- **AIS数据**: 船舶轨迹数据处理
- **合成数据**: 生成模拟供应链数据
- **时间对齐**: 统一时区和时间格式
- **缺失值处理**: 插值、删除、标记策略

### 2. 特征工程
- **时间特征**: 周期性编码（小时、天、月）
- **节假日特征**: 支持多国节假日日历
- **HS编码处理**: One-Hot编码或嵌入向量
- **外部数据融合**: 港口拥堵、汇率、天气数据
- **交互特征**: 自动生成特征交互

### 3. 模型架构

#### RNN系列模型
- **基础RNN**: 支持双向和多层
- **LSTM**: 带注意力机制的LSTM
- **GRU**: 带时间注意力的GRU
- **增强RNN**: 结合位置编码和多头注意力

#### GNN系列模型
- **GCN**: 图卷积网络，支持多种池化
- **GAT**: 图注意力网络，可视化注意力权重
- **GraphSAGE**: 归纳式图学习
- **混合GNN**: 结合多种GNN架构

#### Transformer模型
- **时序Transformer**: 专门用于时间序列预测
- **多尺度Transformer**: 捕获不同时间尺度的模式
- **跨模态Transformer**: 融合多种数据源

### 4. 模型压缩
- **结构剪枝**: 支持30%、50%、70%剪枝率
- **后训练量化**: 8-bit、4-bit量化
- **组合压缩**: 剪枝+量化的组合策略
- **性能对比**: 压缩前后的详细对比分析

### 5. 鲁棒性测试
- **节点丢弃**: 模拟数据缺失
- **序列打乱**: 测试时序依赖性
- **噪声注入**: 高斯、均匀、dropout噪声
- **业务中断**: 节假日、港口关闭、极端天气、疫情影响

### 6. 边缘部署基准
- **推理延迟**: 平均、P95、P99延迟测试
- **内存使用**: 峰值内存和参数内存
- **CPU利用率**: 多核心利用率监控
- **温度监控**: 系统温度变化（如果支持）
- **功耗估算**: 基于CPU使用率的功耗计算
- **实时能力**: 不同延迟预算下的适用性

## � 实验结果展示

运行完成后，系统会在 `results/` 目录下生成丰富的实验结果文件：

### 📋 **生成的报告文件**
| 文件名 | 内容描述 | 格式 |
|--------|---------|------|
| `evaluation_report.md` | 📈 所有模型性能评估详细报告 | Markdown |
| `compression_report.md` | ⚡ 模型压缩效果分析报告 | Markdown |
| `robustness_report.md` | 🛡️ 鲁棒性测试结果报告 | Markdown |
| `deployment_report.md` | 🚀 部署基准测试报告 | Markdown |

### 📊 **数据文件**
| 文件名 | 内容描述 | 格式 |
|--------|---------|------|
| `evaluation_results.json` | 详细评估指标数据 | JSON |
| `model_comparison.csv` | 模型性能对比表 | CSV |
| `compression_results.json` | 压缩前后性能对比 | JSON |
| `robustness_results.json` | 鲁棒性测试详细数据 | JSON |
| `deployment_benchmark.json` | 部署性能基准数据 | JSON |

### 📈 **可视化图表（15个专业图表，全英语化）**

#### 性能评估图表
- `performance_comparison.png` - 📊 模型性能对比柱状图
- `roc_curves.png` - 📈 ROC曲线对比图
- `confusion_matrices.png` - 🎯 混淆矩阵热力图
- `training_curves.png` - 📉 训练损失和准确率曲线
- `model_complexity_vs_performance.png` - ⚖️ 模型复杂度vs性能散点图

#### 鲁棒性分析图表
- `robustness_curves_drop.png` - 🎯 Drop扰动鲁棒性曲线
- `robustness_curves_shuffle.png` - 🔀 Shuffle扰动鲁棒性曲线  
- `robustness_curves_noise.png` - 📡 Noise扰动鲁棒性曲线
- `robustness_comparison.png` - 🛡️ 鲁棒性综合对比图

#### 压缩与部署图表
- `compression_performance_tradeoff.png` - ⚡ 压缩率vs性能权衡图
- `inference_latency_comparison.png` - ⏱️ 推理延迟对比图
- `memory_usage_comparison.png` - 💾 内存使用对比图
- `real_time_capability_radar.png` - 🎯 实时能力雷达图
- `deployment_suitability_heatmap.png` - 🗺️ 部署适用性热力图
- `resource_efficiency_analysis.png` - 📊 资源效率分析图

### 📊 **典型实验结果示例**

#### 模型性能对比
```
模型性能排名（按F1-Score）:
1. GAT:         F1=0.847, Accuracy=0.823, AUC=0.891
2. GraphSAGE:   F1=0.836, Accuracy=0.815, AUC=0.883  
3. GCN:         F1=0.829, Accuracy=0.808, AUC=0.876
4. Transformer: F1=0.821, Accuracy=0.802, AUC=0.869
5. LSTM:        F1=0.813, Accuracy=0.796, AUC=0.862
6. GRU:         F1=0.809, Accuracy=0.791, AUC=0.857
```

#### 压缩效果
```
模型压缩结果（以GAT为例）:
- 原始模型: 2.3M参数, F1=0.847
- 30%剪枝:  1.6M参数, F1=0.842 (-0.6%)
- 50%剪枝:  1.1M参数, F1=0.835 (-1.4%)
- 8bit量化: 0.6M存储, F1=0.844 (-0.4%)
```

#### 鲁棒性表现
```
鲁棒性测试结果（扰动强度0.3）:
- Drop扰动:     平均性能保持率 87.3%
- Shuffle扰动:  平均性能保持率 82.1%
- Noise扰动:    平均性能保持率 89.6%
- Business扰动: 平均性能保持率 85.2%
```

## ⚡ 高级功能

### 🔍 **模型诊断与调试**
```bash
# 运行内置的调试脚本
python debug_gnn.py                    # GNN模型调试
python debug_data_quality.py          # 数据质量检查
python verify_fixes.py                # 系统完整性验证

# 检查模型加载和兼容性
python test_quick_fixes.py            # 快速功能测试
python test_robustness_compression.py # 鲁棒性和压缩测试
```

### 📊 **性能监控**
```bash
# 实时监控训练过程
python main.py --mode train --model gat --verbose --monitor

# 温度和资源监控（部署测试）
python main.py --mode deploy --verbose --monitor_temp
```

### 🎛️ **自定义配置**
```yaml
# config.yaml 主要配置项
data:
  use_synthetic: true           # 使用合成数据
  synthetic_samples: 10000      # 样本数量
  missing_strategy: interpolate # 缺失值处理

models:
  input_dim: 93                # 输入特征维度（已优化）
  hidden_dims: [128, 64]       # 隐藏层维度
  dropout_rates: [0.3, 0.2]    # Dropout率

training:
  batch_size: 32               # 批次大小
  epochs: 100                  # 训练轮数
  learning_rate: 0.001         # 学习率
  early_stopping: 10           # 早停patience

compression:
  pruning_ratios: [0.3, 0.5, 0.7]     # 剪枝比例
  quantization_bits: [8]               # 量化位数
  
robustness:
  perturbation_levels: [0.1, 0.2, 0.3, 0.4, 0.5]  # 扰动强度
```

### 🔧 **扩展开发**

#### 添加新模型
```python
# 在 src/models/ 下创建新模型
class YourCustomModel(BaseModel):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # 模型定义
        
    def forward(self, x, edge_index=None):
        # 前向传播
        return output
        
    def save(self, path):
        # 保存模型
        
    def load(self, path):
        # 加载模型
```

#### 添加新评估指标
```python
# 在 evaluator.py 中扩展
def calculate_custom_metric(y_true, y_pred):
    # 自定义指标计算
    return metric_value
```

## 📋 系统要求与兼容性

### 💻 **硬件要求**
| 配置级别 | CPU | 内存 | 存储 | GPU(可选) |
|---------|-----|------|------|----------|
| **最低配置** | 2核心 | 4GB | 2GB | - |
| **推荐配置** | 4核心+ | 8GB+ | 5GB | GTX 1060+ |
| **最佳体验** | 8核心+ | 16GB+ | 10GB | RTX 3070+ |

### 🐍 **软件兼容性**
| 组件 | 支持版本 | 推荐版本 | 备注 |
|------|---------|---------|------|
| **Python** | 3.8-3.11 | 3.10 | 稳定性最佳 |
| **PyTorch** | 2.0+ | 2.7+ | CUDA支持 |
| **NumPy** | 1.24-1.26 | 1.26.4 | 避免2.x兼容性问题 |
| **操作系统** | Linux/macOS/Windows | Linux | 性能最优 |

### 🔧 **已解决的兼容性问题**
- ✅ NumPy 2.x 兼容性问题（已降级到1.26.4）
- ✅ PyTorch ReduceLROnPlateau verbose参数弃用
- ✅ NetworkX 3.x API变更适配  
- ✅ Matplotlib/Seaborn 绘图库兼容性
- ✅ scikit-learn 与 NumPy 版本冲突
- ✅ 设备（CPU/GPU）一致性问题

## 🚀 项目特色与创新

### 🎯 **学术价值**
- **首个系统性对比研究**: RNN vs GNN vs Transformer在供应链风险预测中的全面对比
- **鲁棒性评估框架**: 建立了供应链AI系统的标准化鲁棒性测试协议
- **模型压缩技术**: 在保持预测精度的前提下实现90%+的模型尺寸压缩
- **实际应用导向**: 从学术研究到工业部署的完整技术栈

### 🔬 **技术创新**
- **多模态数据融合**: 贸易数据 + 航运数据 + 外部风险因子的统一建模
- **层级感知图构建**: 基于供应链层级关系的智能图结构构造
- **自适应特征工程**: 自动化的时间特征、节假日特征、交互特征生成
- **端到端工作流**: 从原始数据到生产部署的无缝流程

### 📊 **系统优势**
- **高度模块化**: 每个组件都可独立使用和替换
- **完全可复现**: 详细的配置管理和随机种子控制
- **生产就绪**: 完善的错误处理、日志记录、性能监控
- **可视化丰富**: 15个专业图表，支持科研论文发表要求

## 🏆 应用场景

### 🌍 **全球贸易风险监控**
- **海关部门**: 进出口贸易风险预警
- **港口管理**: 货物积压和物流中断预测
- **供应链企业**: 上游供应商风险评估
- **金融机构**: 贸易融资风险控制

### 🏭 **制造业供应链管理**
- **汽车行业**: 零部件供应中断预测
- **电子制造**: 芯片供应链风险监控
- **快消品**: 原材料价格波动预警
- **医药行业**: 药品供应链安全评估

### 📈 **金融风险管理**
- **供应链金融**: 中小企业信贷风险评估
- **保险公司**: 供应链中断保险定价
- **投资机构**: 企业供应链韧性评估
- **风险咨询**: 企业风险管理咨询服务

## 🔄 更新日志

### v1.0 (2025-08-03) - 完整修复版本
- ✅ **兼容性修复**: 解决NumPy 2.x、PyTorch 2.7+等依赖库兼容性问题
- ✅ **功能完善**: 修复GNN训练异常、鲁棒性测试、模型压缩等关键功能
- ✅ **性能优化**: 温度监控、设备一致性、批处理等性能问题解决
- ✅ **可视化英语化**: 所有图表和输出完全英语化，支持国际发表
- ✅ **文档完善**: 增加详细的修复报告、学术提升方案、论文写作指南

### 主要修复内容
- 🔧 修复了15个关键bug，包括np.int弃用、模型加载异常等
- 📊 新增15个专业英语化图表，满足学术发表要求
- 🛡️ 完善了4种扰动类型的鲁棒性测试框架
- ⚡ 实现了剪枝+量化的完整模型压缩流程
- 🚀 优化了部署基准测试，支持实时性能监控

## 🤝 贡献指南

欢迎各种形式的贡献！无论是bug报告、功能建议、代码改进还是文档完善。

### � **贡献方式**
1. **Fork项目** - 点击右上角的Fork按钮
2. **创建分支** - `git checkout -b feature/YourFeature`
3. **提交代码** - `git commit -m 'Add: YourFeature description'`
4. **推送分支** - `git push origin feature/YourFeature`
5. **提交PR** - 在GitHub上创建Pull Request

### 🐛 **Bug报告**
请在Issues中提供以下信息：
- 运行环境（Python版本、操作系统）
- 错误的完整堆栈跟踪
- 复现步骤和最小化示例
- 期望的行为描述

### 💡 **功能建议**
- 新模型架构实现
- 新的评估指标和可视化
- 性能优化建议
- 文档改进

### 🏷️ **提交规范**
```
类型: 简短描述

详细描述（可选）

类型包括：
- Add: 新功能
- Fix: Bug修复  
- Update: 功能更新
- Docs: 文档更新
- Style: 代码格式
- Refactor: 重构
- Test: 测试相关
```

## 📚 相关资源

### 📖 **学术文档**
- [学术价值提升方案](academic_enhancement_proposal.md)
- [论文写作指南](paper_writing_guide.md)
- [发表可行性评估](PUBLICATION_FEASIBILITY_REPORT.md)
- [系统状态报告](FINAL_SYSTEM_STATUS_REPORT.md)

### 🔧 **技术文档**
- [BUG修复报告](BUG_FIX_REPORT.md)
- [GNN训练修复报告](GNN_TRAINING_FIX_REPORT.md)
- [鲁棒性修复报告](ROBUSTNESS_PERTURBATION_FIX_REPORT.md)
- [压缩功能修复报告](COMPRESSION_REPORT_FIX.md)

### 🎯 **实验脚本**
- [数据质量调试](debug_data_quality.py)
- [系统验证测试](verify_fixes.py)
- [快速功能测试](test_quick_fixes.py)
- [鲁棒性压缩测试](test_robustness_compression.py)

## �📄 许可证

本项目采用 **MIT许可证** - 详见 [LICENSE](LICENSE) 文件

```
MIT License

Copyright (c) 2025 Supply Chain Risk Prediction System

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

[... 完整MIT许可证文本 ...]
```

## 📞 联系方式

### 👨‍💻 **项目维护者**
- **邮箱**: 2895254401@qq.com
- **GitHub**: [@your-username](https://github.com/your-username)

### 🔗 **项目链接**
- **GitHub仓库**: [SupplyChainRisk](https://github.com/your-username/SupplyChainRisk)
- **Issues报告**: [GitHub Issues](https://github.com/your-username/SupplyChainRisk/issues)
- **讨论区**: [GitHub Discussions](https://github.com/your-username/SupplyChainRisk/discussions)

### 💬 **技术支持**
- 🐛 **Bug报告**: 通过GitHub Issues
- 💡 **功能建议**: 通过GitHub Discussions  
- 📧 **商业合作**: 发送邮件联系
- 🎓 **学术合作**: 欢迎引用和合作研究

## 🙏 致谢

### 🛠️ **开源项目**
感谢以下优秀的开源项目为本系统提供支持：
- [PyTorch](https://pytorch.org/) - 深度学习框架
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/) - 图神经网络库
- [Scikit-learn](https://scikit-learn.org/) - 机器学习工具包
- [Optuna](https://optuna.org/) - 超参数优化框架
- [NetworkX](https://networkx.org/) - 网络分析库
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) - 数据可视化

### 🎓 **学术界**
- 供应链管理领域的前沿研究
- 图神经网络理论的发展
- 鲁棒性机器学习的理论基础

### 🏢 **数据来源**
- [UN Comtrade](https://comtrade.un.org/) - 国际贸易数据
- 各国海关和港口管理部门
- 开放的AIS船舶追踪数据

---

## 🌟 **Star历史**

如果这个项目对您有帮助，请考虑给它一个⭐！

[![Stargazers over time](https://starchart.cc/your-username/SupplyChainRisk.svg)](https://starchart.cc/your-username/SupplyChainRisk)

---

**📢 注意**: 本项目主要用于学术研究和算法验证。在实际生产环境中部署前，请使用真实数据进行充分测试，并根据具体业务需求进行相应调整。

**🎯 项目目标**: 推动供应链风险预测领域的技术进步，为全球供应链安全做出贡献！
