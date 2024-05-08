from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
from thop import profile
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import numpy as np
import argparse
from tqdm import tqdm
import json
from utils.utils import *
from utils.graph import *
from utils.loss import *
from utils.tools import *
from utils.visualize import *
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR
from utils.constants import *
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os

warnings.filterwarnings('ignore')

class Exp_SSL(Exp_Basic):
    def __init__(self, args:argparse.Namespace):
        super(Exp_SSL, self).__init__(args)

    def _build_model(self):
        # model init
        model = self.model_dict[self.args.model].Model(self.args).cuda()
        if self.args.use_gpu:
            if self.args.model in ['DCRNN', 'VQ_MTM', 'SimMTM', 'Ti_MAE', 'BIOT']:
                model = DDP(model, device_ids=[self.device])
            else:
                raise NotImplementedError
        return model
    
    def _get_scalar(self):
        if self.args.normalize:
            means_dir = os.path.join(
                self.args.marker_dir, 'means_seq2seq_nofft.pkl'
                )
            stds_dir = os.path.join(
                self.args.marker_dir, 'stds_seq2seq_nofft.pkl',
                )
            with open(means_dir, 'rb') as f:
                means = pickle.load(f).reshape(-1, 1)
            with open(stds_dir, 'rb') as f:
                stds = pickle.load(f).reshape(-1, 1)
            scalar = StandardScaler(mean=means, std=stds, device=self.device)
        else:
            scalar = None
        return scalar

    def _get_data(self, flag):
        self.args.split = flag
        data_set, data_loader = data_provider(self.args, scalar=self.scalar)
        return data_set, data_loader

    def _select_optimizer(self):
        if self.args.model in ['DCRNN']:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        elif self.args.model in ['VQ_MTM', 'SimMTM', 'Ti_MAE', 'BIOT']:
            params = []
            for name, param in self.model.named_parameters():
                if ('bias' in name) or ('layer_norm' in name) or ('ln' in name):
                    param_group = {'params': [param], 'weight_decay': 0}
                else:
                    param_group = {'params': [param], 'weight_decay': self.args.weight_decay}
                params.append(param_group)
            model_optim = optim.AdamW(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if self.args.model in ['DCRNN']:
            criterion = loss_fn(standard_sclar=None, loss_fn=self.args.loss_fn, is_tensor=True, mask_val=0.)
        elif self.args.model in ['VQ_MTM']:
            criterion = nn.CrossEntropyLoss().to(self.device)
        elif self.args.model in ['SimMTM', 'Ti_MAE', 'BIOT']:
            criterion = None
        return criterion
    
    def _select_scheduler(self, optimizer):
        if self.args.model in ['DCRNN']:
            scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        elif self.args.model in ['VQ_MTM', 'SimMTM', 'Ti_MAE', 'BIOT']:
            scheduler1 = LinearLR(optimizer, start_factor=0.5, total_iters=self.args.warmup_epochs)
            scheduler2 = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs - self.args.warmup_epochs)
            scheduler = SequentialLR(optimizer, [scheduler1, scheduler2], milestones=[self.args.warmup_epochs])
        else:
            raise NotImplementedError
        return scheduler

    def vali(self, vali_loader, criterion):
        losses = []
        self.model.eval()
        with torch.no_grad():
            for x, y, _ in tqdm(vali_loader, disable=(self.device != 0)):
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                # get adjmat, supports
                if self.args.use_graph:
                    _, supports = get_supports(self.args, x)
                else:
                    supports = None

                if self.args.using_patch:
                    batch_size, node_num, seq_len = x.shape
                    x = x.reshape(batch_size, node_num, -1, self.args.freq)
                    x = x.permute(0, 2, 1, 3)

                    y = y.reshape(batch_size, node_num, -1, self.args.freq)
                    y = y.permute(0, 2, 1, 3)

                if self.args.use_fft:
                    x = torch.fft.rfft(x)[..., 1:]
                    x = torch.log(torch.abs(x) + 1e-8)
                    y = torch.fft.rfft(y)[..., 1:]
                    y = torch.log(torch.abs(y) + 1e-8)

                if self.args.model in ['DCRNN']:
                    y_pred = self.model(x, y, supports, None)
                    loss = criterion(y, y_pred)
                elif self.args.model in ['VQ_MTM']:
                    pred, label = self.model(x)
                    loss = self.criterion(pred, label).to(self.device)
                elif self.args.model in ['SimMTM']:
                    loss, _, _, _, _, _, _ = self.model(x)
                elif self.args.model in ['Ti_MAE']:
                    loss, _, _ = self.model(x)
                elif self.args.model in ['BIOT']:
                    loss = self.model(x)
                else:
                    raise NotImplementedError
                loss_val = loss.item()
                losses.append(loss_val)

        loss = np.average(losses)

        self.model.train()
        return loss

    def train(self):
        _, train_loader = self._get_data(flag='train')
        _, vali_loader = self._get_data(flag='dev')

        path = self.logging_dir
        checkpoint_dir = os.path.join(path, 'checkpoint')
        if self.device == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(self.logging_dir, 'graph'), exist_ok=True)

        args_file = os.path.join(self.logging_dir, 'args.json')
        if self.device == 0:
            with open(args_file, 'w') as f:
                json.dump(vars(self.args), f, indent=4, sort_keys=True)

        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        saver = CheckpointSaver(checkpoint_dir, metric_name=self.args.loss_fn,
                                  maximize_metric=False)
        
        if self.args.continue_train:
            if self.device == 0:
                print('continue train')
            self.model, model_optim = load_model_checkpoint(self.args.last_train_path, self.model, model_optim, map_location=self.device)

        self.steps = 0
        for epoch in range(self.args.num_epochs): 
            start_draw = True
            self.model.train()
            train_loader.sampler.set_epoch(epoch)

            with tqdm(train_loader.dataset, desc=f'Epoch: {epoch + 1} / {self.args.num_epochs}', \
                                              disable=(self.device != 0)) as progress_bar:
                for x, y, augment in train_loader:
                    model_optim.zero_grad()

                    batch_size = x.size(0)
                    x = x.float().to(self.device)
                    y = y.float().to(self.device)
                    augment = augment.to(self.device)

                    with torch.no_grad():
                        # get adjmat, supports
                        if self.args.use_graph:
                            x_origin = getOriginalData(x, augment)
                            adj_mat, supports = get_supports(self.args, x_origin)
                        else:
                            supports = None

                        if self.args.using_patch:
                            batch_size, node_num, seq_len = x.shape
                            x = x.reshape(batch_size, node_num, -1, self.args.freq)
                            x = x.permute(0, 2, 1, 3)

                            y = y.reshape(batch_size, node_num, -1, self.args.freq)
                            y = y.permute(0, 2, 1, 3)

                        if self.args.use_fft:
                            x = torch.fft.rfft(x)[..., 1:self.args.input_dim+1]
                            x = torch.log(torch.abs(x) + 1e-8)
                            y = torch.fft.rfft(y)[..., 1:self.args.output_dim+1]
                            y = torch.log(torch.abs(y) + 1e-8)

                        if self.args.use_graph and start_draw:
                            if self.args.graph_type == 'distance' and self.args.adj_every > 0:
                                pos_spec = get_spectral_graph_positions(self.args.marker_dir)
                                fig = draw_graph_weighted_edge(adj_mat, NODE_ID_DICT, pos_spec, title=f'distance_epoch{epoch}.png', 
                                                     is_directed=False, plot_colorbar=True, font_size=30,
                                                     save_dir=os.path.join(self.logging_dir, 'graph'))
                                self.args.adj_every = 0
                                self.logging.add_figure('graph/distance', fig, epoch)
                                start_draw = False
                            elif self.args.graph_type == 'correlation' and (epoch % self.args.adj_every == 0):
                                pos_spec = get_spectral_graph_positions(self.args.marker_dir)
                                fig = draw_graph_weighted_edge(adj_mat, NODE_ID_DICT, pos_spec, title=f'correlation_epoch{epoch}.png', 
                                                     is_directed=self.args.directed, plot_colorbar=True, font_size=30, 
                                                     save_dir=os.path.join(self.logging_dir, 'graph'))
                                self.logging.add_figure(f'graph/correlation_{epoch}', fig, epoch)
                                start_draw = False
                    if self.args.model in ['DCRNN']:
                        seq_pred = self.model(x, y, supports, self.steps)
                        loss = self.criterion(y, seq_pred).to(self.device)
                        loss = self.criterion(pred, label).to(self.device)
                    elif self.args.model in ['VQ_MTM']:
                        pred, label = self.model(x)
                        loss = self.criterion(pred, label).to(self.device)
                    elif self.args.model in ['SimMTM']:
                        loss, _, _, _, _, _, _ = self.model(x)
                    elif self.args.model in ['Ti_MAE']:
                        loss, _, _ = self.model(x)
                    elif self.args.model in ['BIOT']:
                        loss = self.model(x)
                    else:
                        raise NotImplementedError
                    
                    loss_val = loss.item()
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                    self.steps += batch_size * self.world_size
                    model_optim.step()
                    
                    progress_bar.update(batch_size * self.world_size)
                    progress_bar.set_postfix(loss=loss_val, lr=model_optim.param_groups[0]['lr'])

                    
                    self.logging.add_scalar('train/loss', loss_val, self.steps)
                    self.logging.add_scalar('train/lr', model_optim.param_groups[0]['lr'], self.steps)
            

            if (epoch+1) % self.args.eval_every == 0:
                vali_loss = self.vali(vali_loader, self.criterion)
                self.logging.add_scalar('vali/loss', vali_loss, epoch)
                saver.save(epoch, self.model, model_optim, vali_loss)
                early_stopping(vali_loss)
                if early_stopping.early_stop:
                    break
            if self.args.use_scheduler:
                scheduler.step()
        return

    def test(self, model_file:str='best.pth.tar', model_dir:str=None):
        _, test_loader = self._get_data(flag='eval')
        if model_dir is None:
            path = os.path.join(self.logging_dir, 'checkpoint', model_file)
        else:
            path = os.path.join(model_dir, model_file)
        load_model_checkpoint(path, self.model, map_location=self.device, if_strict=True)

        criterion = self._select_criterion()

        self.model.eval()
        losses = []
        with torch.no_grad():
            for x, y, _ in tqdm(test_loader, disable=(self.device != 0)):
                x = x.float().to(self.device)
                y = y.float().to(self.device)

                # get adjmat, supports
                if self.args.use_graph:
                    _, supports = get_supports(self.args, x)
                else:
                    supports = None

                if self.args.using_patch:
                    batch_size, node_num, seq_len = x.shape
                    x = x.reshape(batch_size, node_num, -1, self.args.freq)
                    x = x.permute(0, 2, 1, 3)

                    y = y.reshape(batch_size, node_num, -1, self.args.freq)
                    y = y.permute(0, 2, 1, 3)

                if self.args.use_fft:
                    x = torch.fft.rfft(x)[..., 1:self.args.input_dim+1]
                    x = torch.log(torch.abs(x) + 1e-8)
                    y = torch.fft.rfft(y)[..., 1:self.args.output_dim+1]
                    y = torch.log(torch.abs(y) + 1e-8)

                if self.args.model in ['DCRNN']:
                    y_pred = self.model(x, y, supports, None)
                    loss = criterion(y, y_pred)
                elif self.args.model in ['VQ_MTM']:
                    pred, label = self.model(x)
                    loss = self.criterion(pred, label).to(self.device)
                elif self.args.model in ['SimMTM']:
                    loss, _, _, _, _, _, _ = self.model(x)
                elif self.args.model in ['Ti_MAE']:
                    loss, _, _ = self.model(x)
                elif self.args.model in ['BIOT']:
                    loss = self.model(x)
                else:
                    raise NotImplementedError
                
                loss_val = loss
                losses.append(loss_val)

            loss = torch.mean(torch.stack(losses))
            dist.all_reduce(loss, op=dist.ReduceOp.SUM)
            loss = loss / self.world_size

        self.logging.add_scalar('test/loss', loss)
        return