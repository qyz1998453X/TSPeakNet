# -*- coding: utf-8 -*-
"""
Dual-Branch KAN-TimesNet for Time Series Forecasting
"""

__version__ = "1.0.0"
__author__ = "Qin Yuanze et al."

from .kan import KANLinear, KAN
from .timesnet import KANTimesBlock, FusedTimesNet
from .dual_branch import DualBranchForecast

__all__ = [
    'KANLinear',
    'KAN',
    'KANTimesBlock',
    'FusedTimesNet',
    'DualBranchForecast',
]
