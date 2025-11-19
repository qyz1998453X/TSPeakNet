# -*- coding: utf-8 -*-
"""
Configuration file for Dual-Branch KAN-TimesNet model
"""

import torch

# ==================== Random Seed ====================
SEED = 42

# ==================== Device Configuration ====================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Data Configuration ====================
CHANNELS = [
    "Node_DaXing", "Node_MiYun", "Node_PingGu", "Node_YanQing",
    "Node_HuaiRou", "Node_FangShan", "Node_ChangPing",
    "Node_HaiDian", "Node_TongZhou", "Node_ShunYi"
]

# Data paths
EXCEL_PATH = "../00-时序数据分析/denoised_savgol.xlsx"
PLOT_DIR = "plots"
RESULT_CSV = "02-DB-KFTN-Result.csv"
PREDICTION_XLSX = "02-DB-KFTN-Predictions.xlsx"

# Train/Val/Test split
TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

# ==================== Model Architecture ====================
# Window sizes for dual branches
WINDOW_S = 10  # Short-term branch
WINDOW_L = 30  # Long-term branch

# TimesNet periods
PERIODS = [6, 24]

# Model dimensions
C_IN = 1       # Input channels
C_HID = 64     # Hidden dimension
OUT_LEN = 1    # Output length (1-step ahead)
KS = 3         # Kernel size

# KAN parameters
GRID_SIZE = 5
SPLINE_ORDER = 3
SCALE_NOISE = 0.1
SCALE_BASE = 1.0
SCALE_SPLINE = 1.0
GRID_EPS = 0.02
GRID_RANGE = [-1, 1]

# ==================== Training Configuration ====================
# Offline training
LR = 1e-3
EPOCHS = 150
BATCH_SZ = 128

# Online fine-tuning
ONLINE_STEPS = 3
LR_ONLINE = 1e-4

# Regularization
REG_WEIGHT = 1e-6
EPS_POS = 1e-6

# ==================== Visualization Configuration ====================
# Matplotlib settings
FONT_FAMILY = "SimHei"
FIGURE_SIZE = (12, 4)
DPI = 150
