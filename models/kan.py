# -*- coding: utf-8 -*-
"""
Kolmogorov-Arnold Network (KAN) implementation
Based on: https://github.com/KindXiaoming/pykan
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class KANLinear(nn.Module):
    """
    KAN Linear layer with B-spline basis functions
    """
    def __init__(
            self,
            in_features: int,
            out_features: int,
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            enable_standalone_scale_spline: bool = True,
            base_activation=nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: List[float] = None,
    ):
        super(KANLinear, self).__init__()
        if grid_range is None:
            grid_range = [-1, 1]
            
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Create uniform grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        # Learnable parameters
        self.base_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters"""
        torch.nn.init.kaiming_uniform_(
            self.base_weight,
            a=math.sqrt(5) * self.scale_base
        )
        
        with torch.no_grad():
            noise = (
                (torch.rand(
                    self.grid_size + 1,
                    self.in_features,
                    self.out_features
                ) - 0.5)
                * self.scale_noise
                / self.grid_size
            )
            coeff = self.curve2coeff(
                self.grid.T[self.spline_order: -self.spline_order],
                noise
            )
            factor = (
                1.0
                if self.enable_standalone_scale_spline
                else self.scale_spline
            )
            self.spline_weight.data.copy_(factor * coeff)
            
            if self.enable_standalone_scale_spline:
                torch.nn.init.kaiming_uniform_(
                    self.spline_scaler,
                    a=math.sqrt(5) * self.scale_spline
                )

    def b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute B-spline basis functions
        
        Args:
            x: Input tensor [batch, in_features]
            
        Returns:
            B-spline bases [batch, in_features, grid_size+spline_order]
        """
        grid = self.grid  # [in_features, grid_size+2*spline_order+1]
        x_u = x.unsqueeze(-1)  # [batch, in_features, 1]
        
        # Zeroth order B-splines (step functions)
        bases = ((x_u >= grid[:, :-1]) & (x_u < grid[:, 1:])).to(x_u.dtype)
        
        # Higher order B-splines using Cox-de Boor recursion
        for k in range(1, self.spline_order + 1):
            left = (
                (x_u - grid[:, :-(k+1)]) /
                (grid[:, k:-1] - grid[:, :-(k+1)] + 1e-8)
            ) * bases[:, :, :-1]
            right = (
                (grid[:, k+1:] - x_u) /
                (grid[:, k+1:] - grid[:, 1:(-k)] + 1e-8)
            ) * bases[:, :, 1:]
            bases = left + right
            
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit spline coefficients to curve points
        
        Args:
            x: Grid points [grid_size+1, in_features]
            y: Function values [grid_size+1, in_features, out_features]
            
        Returns:
            Spline coefficients [out_features, in_features, coeffs]
        """
        A = self.b_splines(x).transpose(0, 1)  # [in_features, grid_pts, coeffs]
        B = y.transpose(0, 1)                  # [in_features, grid_pts, out_features]
        sol = torch.linalg.lstsq(A, B).solution  # [in_features, coeffs, out_features]
        return sol.permute(2, 0, 1).contiguous()  # [out_features, in_features, coeffs]

    @property
    def scaled_spline_weight(self) -> torch.Tensor:
        """Get scaled spline weights"""
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(-1)
        return self.spline_weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [..., in_features]
            
        Returns:
            Output tensor [..., out_features]
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        
        # Base activation branch
        base_out = F.linear(self.base_activation(x_flat), self.base_weight)
        
        # Spline branch
        spline_b = self.b_splines(x_flat).view(x_flat.size(0), -1)
        spline_out = F.linear(
            spline_b,
            self.scaled_spline_weight.view(self.out_features, -1)
        )
        
        out = base_out + spline_out
        return out.reshape(*orig_shape[:-1], self.out_features)

    def regularization_loss(
            self,
            regularize_activation: float = 1.0,
            regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """
        Compute regularization loss to encourage sparsity
        
        Args:
            regularize_activation: Weight for activation regularization
            regularize_entropy: Weight for entropy regularization
            
        Returns:
            Regularization loss scalar
        """
        l1 = self.spline_weight.abs().mean(-1)  # [out_features, in_features]
        act = l1.sum()
        p = l1 / (act + 1e-8)
        ent = -torch.sum(p * (p + 1e-8).log())
        
        return (
            regularize_activation * act
            + regularize_entropy * ent
        )


class KAN(nn.Module):
    """
    Multi-layer KAN network
    """
    def __init__(
            self,
            layers_hidden: List[int],
            grid_size: int = 5,
            spline_order: int = 3,
            scale_noise: float = 0.1,
            scale_base: float = 1.0,
            scale_spline: float = 1.0,
            base_activation=nn.SiLU,
            grid_eps: float = 0.02,
            grid_range: List[float] = None,
    ):
        """
        Args:
            layers_hidden: List of layer dimensions [in_dim, hidden1, ..., out_dim]
            grid_size: Number of grid intervals for B-splines
            spline_order: Order of B-spline (3 = cubic)
            scale_noise: Noise scale for initialization
            scale_base: Scale for base weights
            scale_spline: Scale for spline weights
            base_activation: Activation function class
            grid_eps: Grid epsilon
            grid_range: Range for grid initialization
        """
        super(KAN, self).__init__()
        if grid_range is None:
            grid_range = [-1, 1]
            
        self.layers = nn.ModuleList()
        for in_f, out_f in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_f, out_f,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                )
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all KAN layers
        
        Args:
            x: Input tensor [..., input_dim]
            
        Returns:
            Output tensor [..., output_dim]
        """
        for layer in self.layers:
            x = layer(x)
        return x

    def regularization_loss(
            self,
            regularize_activation: float = 1.0,
            regularize_entropy: float = 1.0
    ) -> torch.Tensor:
        """
        Total regularization loss across all layers
        
        Args:
            regularize_activation: Weight for activation regularization
            regularize_entropy: Weight for entropy regularization
            
        Returns:
            Total regularization loss
        """
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )
