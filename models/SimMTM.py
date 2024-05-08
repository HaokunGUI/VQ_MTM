import torch
import torch.nn as nn
import numpy as np
from utils.data_augment import masked_data
from layers.SimMTM_Layer import AutomaticWeightedLoss, AggregationRebuild, ContrastiveWeight, Flatten_Head, Pooler_Head
from layers.Embed import DataEmbedding
import argparse

class SimMTM(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float, kernel_size: int, task_name: str, e_layers: int,
                 freq: int, seq_len: int, in_channel: int, linear_dropout: float, temperature: float, 
                 positive_nums: int, dimension: int, **kwargs):
        super(SimMTM, self).__init__(**kwargs)

        # hyperparameters
        self.task_name = task_name
        self.e_layers = e_layers
        self.in_channel = in_channel
        self.hidden_dim = hidden_dim

        self.enc_embedding = DataEmbedding(
            c_in=freq,
            d_model=hidden_dim,
            max_len=freq * seq_len,
            dropout=dropout,
        )

        # self.encoder = nn.ModuleList([nn.Sequential(
        #         nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, 
        #             bias=False, padding=(kernel_size // 2)),
        #         nn.BatchNorm1d(hidden_dim),
        #         nn.GELU(),
        #         nn.Dropout(dropout)
        #     ) for _ in range(e_layers)]
        # )
        self.encoder = nn.ModuleList([nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=8,
                dim_feedforward=hidden_dim,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(e_layers)])

        if self.task_name =='ssl':
            self.pooling = Pooler_Head(seq_len, hidden_dim, dimension, linear_dropout)
            self.contrastive = ContrastiveWeight(temperature=temperature, 
                                                    positive_nums=positive_nums)
            self.aggregation = AggregationRebuild(temperature=temperature,
                                                 positive_nums=positive_nums)
            self.projection = nn.Linear(hidden_dim, freq)
            self.dropout_ssl = nn.Dropout(linear_dropout)
            self.awl = AutomaticWeightedLoss(2)
            self.mse = torch.nn.MSELoss()

        elif self.task_name == 'anomaly_detection':
            self.decoder_ad = nn.Conv1d(in_channel, 1, kernel_size=1)
            self.linear_dropout = nn.Dropout(linear_dropout)
            self.activation = nn.GELU()
            self.final_projector = nn.Linear(hidden_dim, 1)

        elif self.task_name == 'classification':
            self.decoder_cls = nn.Conv1d(in_channel, 1, kernel_size=1)
            self.linear_dropout = nn.Dropout(linear_dropout)
            self.activation = nn.GELU()
            self.final_projector = nn.Linear(hidden_dim, 4)

        else:
            raise ValueError(f"task_name {self.task_name} is not supported.")
    
    def forward(self, x: torch.Tensor, batch_x: torch.Tensor) -> None:
        bs, seq_len, n_vars = x.shape

        # embedding
        x_enc = self.enc_embedding(x) # [bs x seq_len x d_model]

        # encoder
        for i in range(self.e_layers):
            x_enc = self.encoder[i](x_enc)
        
        if self.task_name == 'ssl':
            p_enc_out = x_enc # [bs x seq_len x d_model]
            s_enc_out = self.pooling(p_enc_out) # [bs x dimension]

            # series weight learning
            loss_cl, similarity_matrix, logits, positives_mask = self.contrastive(s_enc_out)
            rebuild_weight_matrix, agg_enc_out = self.aggregation(similarity_matrix, p_enc_out) 
            agg_enc_out = agg_enc_out.view(bs, seq_len, -1) # [bs x seq_len x d_model]
            # decoder
            agg_enc_out = self.dropout_ssl(agg_enc_out)
            dec_out = self.projection(agg_enc_out) # [bs x seq_len x n_vars]
            pred_batch_x = dec_out[:batch_x.shape[0]]

            loss_rb = self.mse(pred_batch_x, batch_x.detach())
            loss = self.awl(loss_cl, loss_rb)

            return loss, loss_cl, loss_rb, positives_mask, logits, rebuild_weight_matrix, pred_batch_x
        elif self.task_name == 'anomaly_detection':   
            agg_enc_out = x_enc[:batch_x.shape[0]] # [bs x seq_len x d_model]
            agg_enc_out = torch.mean(agg_enc_out, dim=1) # [bs x d_model]
            agg_enc_out = agg_enc_out.view(-1, self.in_channel, self.hidden_dim) # [bs x n_vars x d_model]
            agg_enc_out = self.decoder_ad(agg_enc_out).squeeze(1) # [bs x d_model]
            agg_enc_out = self.linear_dropout(agg_enc_out)
            agg_enc_out = self.activation(agg_enc_out)
            enc_out = self.final_projector(agg_enc_out)
            return enc_out
        elif self.task_name == 'classification':
            agg_enc_out = x_enc[:batch_x.shape[0]] # [bs x seq_len x d_model]
            agg_enc_out = torch.mean(agg_enc_out, dim=1) # [bs x d_model]
            agg_enc_out = agg_enc_out.view(-1, self.in_channel, self.hidden_dim)
            agg_enc_out = self.decoder_cls(agg_enc_out).squeeze(1)
            agg_enc_out = self.linear_dropout(agg_enc_out)
            agg_enc_out = self.activation(agg_enc_out)
            enc_out = self.final_projector(agg_enc_out)
        else:
            raise ValueError(f"task_name {self.task_name} is not supported.")
        

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()

        self.args = args

        self.model = SimMTM(
            hidden_dim=args.d_model,
            dropout=args.dropout,
            e_layers=args.e_layers,
            task_name=args.task_name,
            in_channel=args.num_nodes,
            freq=args.freq,
            seq_len=args.input_len,
            linear_dropout=args.linear_dropout,
            temperature=args.temperature,
            positive_nums=args.positive_nums,
            dimension=args.dimension,
            kernel_size=args.kernel_size,
        )
    
    def forward(self, x:torch.Tensor):
        # x: [bs x n_vars x seq_len]
        x = x.view(-1, self.args.input_len, self.args.freq) # [bs*n_vars x seq_len x dim]

        # permute to [bs x dim x seq_len]
        # x = x.permute(0, 2, 1)
        # data augmentation
        batch_x_mark = torch.ones_like(x)
        batch_x_m, _, _ = masked_data(x.cpu(), batch_x_mark.cpu(), self.args.mask_ratio, self.args.lm, self.args.positive_nums)
        batch_x_m = batch_x_m.cuda()
        batch_x_om = torch.cat([x, batch_x_m], dim=0)

        if self.args.task_name == 'ssl':
            return self.model(batch_x_om, x)
        elif self.args.task_name == 'anomaly_detection':
            return self.model(batch_x_om, x)
        elif self.args.task_name == 'classification':
            return self.model(batch_x_om, x)
        else:
            raise ValueError(f"task_name {self.task_name} is not supported.")
