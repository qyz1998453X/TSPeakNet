# -*- coding: utf-8 -*-
"""
Dual-Branch Fusion Model
Combines short-term and long-term branches with learnable weights
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .timesnet import FusedTimesNet


class DualBranchForecast(nn.Module):
    """
    Dual-branch forecasting model
    
    Architecture:
        - Branch S: Short-term (window_s steps)
        - Branch L: Long-term (window_l steps)
        - Learnable fusion weights (softmax normalized)
    """
    
    def __init__(
            self,
            c_in: int = 1,
            c_hid: int = 64,
            periods: list = None,
            out_len: int = 1,
            kernel_size: int = 3
    ):
        """
        Args:
            c_in: Input channels
            c_hid: Hidden dimension
            periods: List of periods for TimesNet
            out_len: Output length
            kernel_size: Convolution kernel size
        """
        super().__init__()
        
        if periods is None:
            periods = [6, 24]
            
        # Short-term branch
        self.net_s = FusedTimesNet(c_in, c_hid, periods, out_len, kernel_size)
        
        # Long-term branch
        self.net_l = FusedTimesNet(c_in, c_hid, periods, out_len, kernel_size)
        
        # Learnable fusion weights
        self.w = nn.Parameter(torch.randn(2))

    def forward(self, xs: torch.Tensor, xl: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            xs: Short-term input [B, C, window_s]
            xl: Long-term input [B, C, window_l]
            
        Returns:
            Fused prediction [B]
        """
        # Get predictions from both branches
        ps = self.net_s(xs)[:, 0, 0]  # [B]
        pl = self.net_l(xl)[:, 0, 0]  # [B]
        
        # Normalize fusion weights with softmax
        w = F.softmax(self.w, dim=0)
        
        # Weighted combination
        return w[0] * ps + w[1] * pl
    
    def get_fusion_weights(self) -> torch.Tensor:
        """Get current fusion weights"""
        return F.softmax(self.w, dim=0)
    
    def get_branch_predictions(
            self,
            xs: torch.Tensor,
            xl: torch.Tensor
    ) -> tuple:
        """
        Get individual branch predictions
        
        Args:
            xs: Short-term input
            xl: Long-term input
            
        Returns:
            Tuple of (short_pred, long_pred, fused_pred)
        """
        ps = self.net_s(xs)[:, 0, 0]
        pl = self.net_l(xl)[:, 0, 0]
        w = F.softmax(self.w, dim=0)
        pf = w[0] * ps + w[1] * pl
        
        return ps, pl, pf
