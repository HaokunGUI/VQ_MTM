import torch 
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class Trans_Conv(nn.Module):
    def __init__(self, d_model: int, dropout: float, in_channels: int, out_channels: int, activation: str='gelu',
                 **kwargs):
        super(Trans_Conv, self).__init__(**kwargs)

        self.transformer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8,
            dim_feedforward=4*d_model,
            dropout=dropout,
            activation=activation,
            batch_first=True,
        )

    
    def forward(self, x: torch.Tensor):
        # x: (B, C, T, d_model)
        B, C, T, D = x.shape

        # get the feature
        x = x.contiguous().view(B*C, T, D)
        x = self.transformer(x)
        x = x.view(B, C, T, D) # (B, C, T, d_model)

        return x
