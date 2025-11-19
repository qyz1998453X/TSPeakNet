# -*- coding: utf-8 -*-
"""
Dataset classes for time series forecasting
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """
    Sliding window dataset for time series
    """
    
    def __init__(self, series: np.ndarray, window: int):
        """
        Args:
            series: 1D time series array
            window: Window size
        """
        self.X = []
        self.y = []
        
        # Create sliding windows
        for i in range(len(series) - window):
            self.X.append(series[i:i+window])
            self.y.append(series[i+window])
        
        # Convert to tensors
        self.X = torch.tensor(np.stack(self.X), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            Tuple of (input_window, target_value)
        """
        return self.X[idx], self.y[idx]


class DualWindowDataset(Dataset):
    """
    Dataset with two window sizes for dual-branch model
    """
    
    def __init__(
            self,
            series: np.ndarray,
            window_s: int,
            window_l: int
    ):
        """
        Args:
            series: 1D time series array
            window_s: Short window size
            window_l: Long window size
        """
        assert window_l > window_s, "Long window must be larger than short window"
        
        self.window_s = window_s
        self.window_l = window_l
        
        self.X_s = []
        self.X_l = []
        self.y = []
        
        # Create dual windows
        for i in range(len(series) - window_l):
            self.X_l.append(series[i:i+window_l])
            self.X_s.append(series[i+window_l-window_s:i+window_l])
            self.y.append(series[i+window_l])
        
        # Convert to tensors
        self.X_s = torch.tensor(np.stack(self.X_s), dtype=torch.float32)
        self.X_l = torch.tensor(np.stack(self.X_l), dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            Tuple of (short_window, long_window, target_value)
        """
        return self.X_s[idx], self.X_l[idx], self.y[idx]
