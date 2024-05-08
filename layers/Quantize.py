import torch
import torch.nn as nn
import torch.nn.functional as F

class Quantize(nn.Module):
    def __init__(self, input_dim: int, vq_dim: int, num_embed: int, codebook_num: int=1, split_num:int=4, **kwargs):
        super(Quantize, self).__init__(**kwargs)
        # hyper-parameters
        self.split_num = split_num

        self.projector = torch.nn.init.xavier_normal_(torch.empty(input_dim, vq_dim))
        self.projector = nn.Parameter(self.projector, requires_grad=False)

        codebook = torch.nn.init.normal_(torch.empty(codebook_num, num_embed, vq_dim))
        self.codebook = nn.Parameter(codebook, requires_grad=False)

        self.linear = torch.arange(input_dim//2 + 1).cuda()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x [bs, T, input_dim]
        with torch.no_grad():
            x_fft = torch.fft.rfft(x, dim=-1)
            magnitude = torch.abs(x_fft)
            phase = torch.angle(x_fft)
            phase -= phase[:, :, 1:2] * self.linear
            x_recon = magnitude * torch.exp(1j * phase)
            x_recon = torch.fft.irfft(x_recon, dim=-1).unsqueeze(-2) # [bs, T, 1, input_dim]

            x_feature = x_recon.matmul(self.projector) # [bs, T, 1, vq_dim]

            x_feature_norm = x_feature.norm(dim=-1, keepdim=True)
            x_feature = x_feature / x_feature_norm # [bs, T, 1, vq_dim]
            codebook_norm = self.codebook.norm(dim=-1, keepdim=True)
            codebook = self.codebook / codebook_norm # [codebook_num, num_embed, vq_dim]

            similarity = torch.einsum('btnd,cmd->btncm', x_feature, codebook) # [bs, T, 1, codebook_num, num_embed]
            idx = torch.argmax(similarity, dim=-1) # [bs, T, 1, codebook_num]
            return idx
