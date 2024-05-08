import torch
import torch.nn as nn

class Patch_Embed(nn.Module):
    def __init__(self, in_channel: int, embed_dim: int, patch_size: int, bias: bool=True, **kwargs):
        super(Patch_Embed, self).__init__(**kwargs)

        self.proj = nn.Conv1d(
            in_channels=in_channel,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=bias,
        )

        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x) # (B, embed_dim, patch_num)
        x = x.transpose(1, 2)
        x = self.norm(x) # (B, patch_num, embed_dim)
        return x