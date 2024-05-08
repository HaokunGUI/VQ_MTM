import torch
import torch.nn as nn
import argparse

class iTransformer(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, linear_dropout: float, 
                 e_layers: int, task_name: str, in_channel: int, activation: str='gelu', **kwargs):
        super(iTransformer, self).__init__(**kwargs)
        
        self.task_name = task_name

        self.embedding = nn.Linear(in_channel, d_model)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_head,
                dropout=dropout,
                activation=activation,
                batch_first=True,
            ),
            num_layers=e_layers,
        )

        if self.task_name == 'anomaly_detection':
            self.agg = nn.Conv1d(in_channels=in_channel, 
                                 out_channels=1, 
                                 kernel_size=1
                                )
            self.activation = self._get_activation_fn(activation)
            self.dropout = nn.Dropout(linear_dropout)
            self.decoder_ad = nn.Linear(d_model, 1)
            
        else:
            raise RuntimeError(f"task_name should be ssl/anomaly_detection, not {self.task_name}")
        
    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        B, T, C = x.shape
        x = x.transpose(1, 2) # (B, C, T)
        x = self.embedding(x) # (B, C, d_model)
        y = self.encoder(x)  # (B, C, d_model)

        if self.task_name == 'anomaly_detection':
            y = self.agg(y).squeeze(1) # (B, d_model)
            y = self.activation(y)
            y = self.dropout(y)
            y = self.decoder_ad(y) # (B, 1)
            return y
        else:
            raise RuntimeError(f"task_name should be ssl/anomaly_detection, not {self.task_name}")
        

class Model(nn.Module):
    def __init__(self, args: argparse.Namespace):
        super(Model, self).__init__()
        self.args = args
        self.model = iTransformer(
            d_model=args.d_model,
            n_head=args.n_head,
            dropout=args.dropout,
            linear_dropout=args.linear_dropout,
            e_layers=args.e_layers,
            task_name=args.task_name,
            in_channel=args.num_nodes,
            activation=args.activation,
        )
    
    def forward(self, x: torch.Tensor):
        # x: (B, T, C)
        return self.model(x)