# -*- coding: utf-8 -*-
"""
TimesNet modules with KAN integration
Reference: TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from .kan import KANLinear


class KANTimesBlock(nn.Module):
    """
    TimesBlock with KAN-based transformation
    Performs 2D convolutions on period-reshaped sequences
    """
    def __init__(self, c_in: int, c_hid: int, period: int, kernel_size: int = 3):
        """
        Args:
            c_in: Input channels
            c_hid: Hidden channels
            period: Period for 2D reshaping
            kernel_size: Convolution kernel size
        """
        super().__init__()
        self.P = period
        pad = kernel_size // 2
        
        # Three types of convolutions for capturing different patterns
        self.conv1 = nn.Conv2d(c_in, c_hid, (1, kernel_size), padding=(0, pad))
        self.conv2 = nn.Conv2d(c_in, c_hid, (kernel_size, 1), padding=(pad, 0))
        self.conv3 = nn.Conv2d(c_in, c_hid, (kernel_size, kernel_size), padding=pad)
        
        # KAN layer for adaptive transformation
        self.kan = KANLinear(c_hid, c_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            Output tensor [B, C, L]
        """
        B, C, L0 = x.size()
        
        # Pad sequence to be divisible by period
        if L0 % self.P:
            L = ((L0 + self.P - 1) // self.P) * self.P
            x_p = F.pad(x, (0, L - L0), mode='replicate')
        else:
            x_p, L = x, L0
            
        # Reshape to 2D: [B, C, num_periods, period]
        x2 = x_p.view(B, C, L // self.P, self.P)
        
        # Apply convolutions
        y = self.conv1(x2) + self.conv2(x2) + self.conv3(x2)
        
        # Reshape back to 1D
        y = y.view(B, -1, L)
        
        # Apply KAN transformation
        y_t = y.transpose(1, 2)              # [B, L, c_hid]
        y_f = self.kan(y_t).transpose(1, 2)  # [B, c_in, L]
        
        # Residual connection and trim to original length
        return (y_f + x_p)[:, :, :L0]


class FusedTimesNet(nn.Module):
    """
    Multi-period TimesNet with fusion
    """
    def __init__(
            self,
            c_in: int,
            c_hid: int,
            periods: List[int],
            out_len: int,
            kernel_size: int = 3
    ):
        """
        Args:
            c_in: Input channels (typically 1 for univariate)
            c_hid: Hidden dimension
            periods: List of periods to capture
            out_len: Output sequence length
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        # Create a TimesBlock for each period
        self.blocks = nn.ModuleList([
            KANTimesBlock(c_in, c_hid, P, kernel_size)
            for P in periods
        ])
        
        # Fusion layer: mix outputs from different periods
        self.mixer = nn.Conv1d(len(periods) * c_in, c_in, 1)
        
        # Projection head for final prediction
        self.head = nn.Conv1d(c_in, c_in * out_len, 1)
        self.out_len = out_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, L]
            
        Returns:
            Output tensor [B, C, out_len]
        """
        # Process each period branch
        outs = [blk(x) for blk in self.blocks]
        
        # Concatenate all branches
        cat = torch.cat(outs, dim=1)  # [B, len(periods)*c_in, L]
        
        # Mix branches
        mix = self.mixer(cat)  # [B, c_in, L]
        
        # Project to output
        y = self.head(mix)  # [B, c_in*out_len, L]
        
        # Take the last time step
        last = y[..., -1]  # [B, c_in*out_len]
        
        # Reshape to [B, c_in, out_len]
        return last.view(-1, x.size(1), self.out_len)
