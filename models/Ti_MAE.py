import torch
import torch.nn as nn
import argparse

from layers.Ti_MAE_Layer import Patch_Embed
from layers.Embed import PositionalEmbedding

class Ti_MAE(nn.Module):
    def __init__(self, series_len: int, patch_size: int, in_chans: int, embed_dim: int, mask_ratio: float, 
                 n_head: int, dropout: float, e_layers: int, d_layers: int, task_name: str, decoder_embed_dim: int, 
                 linear_dropout: int, activation: str='gelu', global_pool: bool=False, **kwargs):
        super(Ti_MAE, self).__init__(**kwargs)

        # hyper-parameters
        self.task_name = task_name
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.global_pool = global_pool

        self.patch_embed = Patch_Embed(
            in_channel=in_chans,
            embed_dim=embed_dim,
            patch_size=patch_size,
        )
        
        self.pos_embed = PositionalEmbedding(
            d_model=embed_dim,
            max_len=60*250//patch_size+1,
        )

        self.cls_token = nn.Parameter(torch.normal(0, 0.1, (1, 1, embed_dim)))
        self.mask_token = nn.Parameter(torch.normal(0, 0.1, (1, 1, decoder_embed_dim)))

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=n_head,
                dim_feedforward=4*embed_dim,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=e_layers,
        )

        if self.task_name == 'ssl':
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.decoder_pos_embed = PositionalEmbedding(
                d_model=decoder_embed_dim,
                max_len=series_len//patch_size+1,
            )

            self.decoder = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=decoder_embed_dim,
                    nhead=n_head,
                    dim_feedforward=4*decoder_embed_dim,
                    dropout=dropout,
                    activation=activation,
                    batch_first=True,
                ),
                num_layers=d_layers,
            )

            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size*in_chans, bias=True)
        elif self.task_name == 'anomaly_detection':
            self.pos_dropout = nn.Dropout(p=linear_dropout)
            self.final_projector = nn.Linear(embed_dim, 1, bias=True)

        elif self.task_name == 'classification':
            self.pos_dropout = nn.Dropout(p=linear_dropout)
            self.final_projector = nn.Linear(embed_dim, 4, bias=True)


    def forward(self, x):
        if self.task_name == 'ssl':
            latent, mask, ids_restore = self.forward_encoder(x, mask_ratio=self.mask_ratio)
            pred = self.forward_decoder(latent, ids_restore)
            loss = self.forward_loss(x, pred, mask)
            return loss, pred, mask
        elif self.task_name == 'anomaly_detection':
            outcome = self.forward_feature(x)
            return outcome
        elif self.task_name == 'classification':
            outcome = self.forward_feature(x)
            return outcome
        else:
            raise NotImplementedError

    def forward_encoder(self, x: torch.Tensor, mask_ratio: float) -> ...:
        # x: (B, in_chans, series_len)
        # embed_patch
        x = self.patch_embed(x) # (B, patch_num, embed_dim)
        # add pos embed w/o cls
        pos_embed = self.pos_embed(torch.zeros(x.shape[0], x.shape[1]+1, x.shape[2], device=x.device))
        x = x + pos_embed[:, 1:, :] # (B, patch_num, embed_dim)
        # random masking
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        # append cls token
        cls_token = self.cls_token + pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, patch_num+1, embed_dim)

        # apply the transformer blocks
        x = self.encoder(x)
        
        return x, mask, ids_restore
    
    def forward_decoder(self, x: torch.Tensor, ids_restore: torch.Tensor) -> ...:
        # x: (B, patch_num+1, embed_dim)
        # embed_tokens
        x = self.decoder_embed(x) # (B, patch_num+1, decoder_embed_dim)
        # append the mask token to sequence
        mask_token = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_token], dim=1) # no cls toekn
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])) # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1) # (B, patch_num+1, decoder_embed_dim)

        # add pos embed
        x = x + self.decoder_pos_embed(x)
        # apply the transformer blocks
        x = self.decoder(x)
        # predict the masked tokens
        x = self.decoder_pred(x[:, 1:, :])

        return x
    
    def forward_loss(self, x: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor):
        # x: (B, in_chans, series_len)
        B, C, L = x.shape
        target = x.transpose(1, 2) # (B, series_len, in_chans)
        target = target.contiguous().view(B, L//self.patch_size, self.patch_size*C)

        loss = (pred - target)**2
        loss = loss.mean(dim=-1) # (B, patch_num)
        loss = (loss * mask).sum() / mask.sum() # (B, )
        return loss
    
    def forward_feature(self, x: torch.Tensor):
        B = x.shape[0]
        x = self.patch_embed(x) # (B, patch_num, embed_dim)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, patch_num+1, embed_dim)
        x = x + self.pos_embed(x) # (B, patch_num, embed_dim)
        x = self.pos_dropout(x)
        x = self.encoder(x)
        
        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.final_projector(x)
        else:
            x = x[:, 0]
            outcome = self.final_projector(x)
        return outcome

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore


class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.model = Ti_MAE(
            series_len=args.input_len*args.freq,
            patch_size=args.freq,
            in_chans=args.num_nodes,
            embed_dim=args.d_model,
            mask_ratio=args.mask_ratio,
            n_head=8,
            dropout=args.dropout,
            linear_dropout=args.linear_dropout,
            e_layers=args.e_layers,
            d_layers=args.d_layers,
            task_name=args.task_name,
            decoder_embed_dim=args.d_model,
            activation=args.activation,
            global_pool=args.global_pool,
        )
    
    def forward(self, x:torch.Tensor):
        return self.model(x)