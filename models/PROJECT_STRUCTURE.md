# 项目结构说明

## 📁 完整目录结构

```
models/                                 # 模型代码根目录
│
├── config.py                          # 全局配置文件
│   ├── 随机种子、设备配置
│   ├── 数据路径和通道定义
│   ├── 模型架构参数
│   ├── 训练超参数
│   └── 可视化配置
│
├── kan.py                             # KAN网络实现
│   ├── KANLinear                      # KAN线性层（B样条基函数）
│   └── KAN                            # 多层KAN网络
│
├── timesnet.py                        # TimesNet模块
│   ├── KANTimesBlock                  # 集成KAN的TimesBlock
│   └── FusedTimesNet                  # 多周期融合TimesNet
│
├── dual_branch.py                     # 双分支融合模型
│   └── DualBranchForecast             # 主预测模型
│       ├── net_s                      # 短期分支 (window=10)
│       ├── net_l                      # 长期分支 (window=30)
│       └── fusion weights             # 可学习融合权重
│
├── train.py                           # 训练主脚本
│   ├── set_seed()                     # 设置随机种子
│   ├── train_model()                  # 模型训练
│   ├── online_finetune()              # 在线微调
│   ├── rolling_forecast()             # 滚动预测
│   ├── train_and_evaluate_channel()   # 单通道完整流程
│   └── main()                         # 主函数
│
├── data/                              # 数据处理模块
│   ├── __init__.py
│   ├── dataset.py                     # 数据集类
│   │   ├── WindowDataset              # 单窗口数据集
│   │   └── DualWindowDataset          # 双窗口数据集
│   └── dataloader.py                  # 数据加载器
│       ├── TimeSeriesDataLoader       # 时序数据加载类
│       │   ├── load_data()            # 加载Excel数据
│       │   ├── fit_scalers()          # 拟合标准化器
│       │   ├── transform_data()       # 数据标准化
│       │   └── inverse_transform()    # 反标准化
│       └── create_dual_dataloaders()  # 创建双分支数据加载器
│
├── utils/                             # 工具函数模块
│   ├── __init__.py
│   ├── metrics.py                     # 评估指标
│   │   ├── safe_mape()                # 安全MAPE计算
│   │   ├── calculate_metrics()        # 综合指标计算
│   │   ├── print_metrics()            # 打印指标
│   │   └── aggregate_metrics()        # 聚合多通道指标
│   └── visualization.py               # 可视化工具
│       ├── setup_matplotlib()         # 配置matplotlib
│       ├── plot_forecast()            # 绘制预测曲线
│       ├── plot_multi_channel()       # 多通道绘图
│       ├── plot_training_curve()      # 训练曲线
│       └── plot_metrics_comparison()  # 指标对比图
│
├── requirements.txt                   # Python依赖
├── README.md                          # 项目说明文档
├── LICENSE                            # MIT开源协议
├── .gitignore                         # Git忽略规则
└── PROJECT_STRUCTURE.md               # 本文件

```

## 🔄 数据流程

```
原始Excel数据 (denoised_savgol.xlsx)
        ↓
TimeSeriesDataLoader.load_data()
        ↓
Train/Val/Test Split (60%/20%/20%)
        ↓
TimeSeriesDataLoader.fit_scalers()
        ↓
Z-score Normalization
        ↓
WindowDataset (window_s=10, window_l=30)
        ↓
DataLoader (batch_size=128)
        ↓
DualBranchForecast Model
        ↓
Training (150 epochs)
        ↓
Rolling Forecast + Online Fine-Tuning
        ↓
Inverse Transform
        ↓
Metrics Calculation & Visualization
        ↓
Results (CSV + Excel + Plots)
```

## 🧠 模型架构

```
输入时序数据 [B, 1, L]
        │
        ├──────────────┬──────────────┐
        │              │              │
   Short Branch    Long Branch       │
   (10 steps)     (30 steps)         │
        │              │              │
  FusedTimesNet   FusedTimesNet      │
   periods=[6,24]  periods=[6,24]    │
        │              │              │
    ┌───┴───┐      ┌───┴───┐         │
    │Block 1│      │Block 1│         │
    │Block 2│      │Block 2│         │
    └───┬───┘      └───┬───┘         │
        │              │              │
   Conv Mixer     Conv Mixer         │
        │              │              │
   KAN Transform  KAN Transform      │
        │              │              │
        └──────┬───────┘              │
               │                      │
       Learnable Fusion              │
       w = softmax([w_s, w_l])       │
       pred = w[0]*p_s + w[1]*p_l    │
               │                      │
           One-Step Pred              │
               │                      │
      Online Fine-Tuning ←───────────┘
      (SGD, 3 steps, lr=1e-4)
               │
          Final Output
```

## 📊 训练流程

### 1. 离线训练阶段

```python
for epoch in range(150):
    for (xs, ys), (xl, yl) in zip(dataloader_s, dataloader_l):
        # Forward
        pred = model(xs, xl)
        
        # Loss = MSE + KAN Regularization
        loss = MSE(pred, ys) + λ * (reg_s + reg_l)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

### 2. 在线微调阶段

```python
for t in test_period:
    # 1. Predict
    pred_t = model(x_s[-10:], x_l[-30:])
    
    # 2. Fine-tune with true value
    for _ in range(3):
        loss = MSE(model(x_s, x_l), y_true_t)
        sgd.zero_grad()
        loss.backward()
        sgd.step()
    
    # 3. Update history
    history.append(y_true_t)
```

## 🎯 核心组件说明

### KANLinear

- **功能**: 使用B样条基函数的可学习非线性变换
- **输入**: `[..., in_features]`
- **输出**: `[..., out_features]`
- **参数**: 
  - `base_weight`: 基础线性权重
  - `spline_weight`: 样条权重
  - `grid`: B样条网格点

### KANTimesBlock

- **功能**: 周期性2D卷积 + KAN变换
- **输入**: `[B, C, L]` (时序数据)
- **输出**: `[B, C, L]` (变换后)
- **操作**:
  1. 按周期重塑为2D: `[B, C, L/P, P]`
  2. 三种卷积: 行、列、2D
  3. KAN变换
  4. 残差连接

### FusedTimesNet

- **功能**: 多周期TimesBlock融合
- **输入**: `[B, C, L]`
- **输出**: `[B, C, out_len]`
- **周期**: `[6, 24]` (周和日周期)

### DualBranchForecast

- **功能**: 双分支预测 + 自适应融合
- **输入**: `xs [B,1,10]`, `xl [B,1,30]`
- **输出**: `[B]` (融合预测)
- **融合**: `pred = softmax(w)[0] * pred_s + softmax(w)[1] * pred_l`

## 📈 输出文件

### 1. 02-DB-KFTN-Result.csv
```csv
Node,MAE,RMSE,R2,MAPE
Node_DaXing,1.23,1.87,0.91,8.45
Node_MiYun,1.45,2.12,0.89,10.23
...
```

### 2. 02-DB-KFTN-Predictions.xlsx
```
Date          | Node_DaXing | Node_MiYun | ...
2021-01-01    | 12.34       | 15.67      | ...
2021-01-02    | 13.45       | 16.78      | ...
...
```

### 3. plots/Node_XXX.png
- 完整时序数据 (黑色实线)
- 预测结果 (橙色虚线)
- 标题、图例、网格

## 🔧 配置参数

### 关键超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `WINDOW_S` | 10 | 短期窗口大小 |
| `WINDOW_L` | 30 | 长期窗口大小 |
| `PERIODS` | [6, 24] | TimesNet周期 |
| `C_HID` | 64 | 隐藏层维度 |
| `LR` | 1e-3 | 学习率 |
| `EPOCHS` | 150 | 训练轮数 |
| `BATCH_SZ` | 128 | 批大小 |
| `ONLINE_STEPS` | 3 | 在线微调步数 |
| `LR_ONLINE` | 1e-4 | 在线学习率 |
| `REG_WEIGHT` | 1e-6 | KAN正则化权重 |

## 🚀 快速使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置数据路径
# 编辑 config.py 中的 EXCEL_PATH

# 3. 运行训练
python train.py

# 4. 查看结果
# - CSV: TSPeakNet-Result.csv
# - Excel: TSPeakNet-Predictions.xlsx
# - Plots: plots/*.png
```

## 📚 参考文献

1. **TimesNet**: Wu et al., "TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis", ICLR 2023
2. **KAN**: Liu et al., "KAN: Kolmogorov-Arnold Networks", arXiv 2024
3. **双分支架构**: 本项目提出的创新架构

## 💡 设计理念

1. **模块化**: 每个组件职责清晰，易于维护和扩展
2. **可复现**: 固定随机种子，详细文档
3. **学术标准**: 符合深度学习论文代码发布规范
4. **易用性**: 配置集中，接口简洁
5. **可扩展**: 易于添加新模型、新数据集