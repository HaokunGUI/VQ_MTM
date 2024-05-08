from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping
import torch
import torch.nn as nn
from torch import optim
import os
import warnings
import argparse
from tqdm import tqdm
import json
from utils.utils import *
from utils.graph import *
from utils.loss import *
from utils.tools import *
from utils.visualize import *
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.constants import *
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import barrier
import os
import torch.distributed as dist
from torchmetrics.classification import MulticlassF1Score, MulticlassAccuracy, MulticlassConfusionMatrix


warnings.filterwarnings('ignore')

class Exp_Classification(Exp_Basic):
    def __init__(self, args:argparse.Namespace):
        super(Exp_Classification, self).__init__(args)
        self.f1 = MulticlassF1Score(num_classes=4, average='weighted').to(self.device)
        self.acc = MulticlassAccuracy(average='weighted', num_classes=4).to(self.device)

    def _build_model(self):
        # model init
        model = self.model_dict[self.args.model].Model(self.args).cuda()
        if self.args.use_gpu:
            if self.args.model in ['DCRNN', 'TimesNet', 'VQ_MTM', 'Ti_MAE']:
                model = nn.DataParallel(model, device_ids=[self.device])
            else:
                raise NotImplementedError
        if self.args.use_pretrained:
            load_model_checkpoint(self.args.pretrained_path, model, map_location=self.device)

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
        if self.args.model in ['VQ_MTM', 'Ti_MAE']:
            params = []
            for name, param in self.model.named_parameters():
                # only update the parameters that are not frozen
                if not param.requires_grad:
                    continue
                # not using decay in bias & ln
                if any([f in name for f in ['bias', 'layer_norm', 'ln']]):
                    weight_decay = 0.0
                else:
                    weight_decay = self.args.weight_decay
                    
                if any([f in name for f in ['final_projector_cls', 'decoder_cls', 'final_projector']]):
                    param_group = {'params': param, 'lr': self.args.learning_rate, 'weight_decay': weight_decay}
                    params.append(param_group)
                else:
                    param_group = {'params': param, 'lr': self.args.learning_rate, 'weight_decay': weight_decay}
                    params.append(param_group)
                
            model_optim = optim.AdamW(params)
        else:
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        return model_optim

    def _select_criterion(self):
        if self.args.model in ['DCRNN', 'VQ_MTM', 'Ti_MAE']:
            criterion = nn.CrossEntropyLoss().cuda()
        elif self.args.model in ['TimesNet']:
            criterion = nn.CrossEntropyLoss().cuda()
        else:
            raise NotImplementedError
        return criterion
    
    def _select_scheduler(self, optimizer):
        scheduler = CosineAnnealingLR(optimizer, T_max=self.args.num_epochs)
        return scheduler

    def vali(self):
        _, vali_loader = self._get_data(flag='dev')

        y_preds = []
        y_trues = []
        self.model.eval()

        with torch.no_grad():
            for x, y, _ in tqdm(vali_loader, disable=(self.device != 0)):
                x = x.float().to(self.device)
                y = y.to(self.device)

                # get adjmat, supports
                if self.args.use_graph:
                    _, supports = get_supports(self.args, x)
                else:
                    supports = None

                if self.args.using_patch:
                    batch_size, node_num, seq_len = x.shape
                    x = x.reshape(batch_size, node_num, -1, self.args.freq)
                    x = x.permute(0, 2, 1, 3)

                if self.args.use_fft:
                    x = torch.fft.rfft(x)[..., 1:]
                    x = torch.log(torch.abs(x) + 1e-8)

                if self.args.model in ['DCRNN']:
                    seq_len = torch.ones(x.shape[0], dtype=torch.int64).cuda() * self.args.input_len
                    y_pred = self.model(x, seq_len, supports)
                elif self.args.model in ['TimesNet']:
                    y_pred = self.model(x)
                elif self.args.model in ['VQ_MTM', 'Ti_MAE']:
                    y_pred = self.model(x)
                else:
                    raise NotImplementedError

                y_preds.append(y_pred)
                y_trues.append(y)

        y_pred = torch.cat(y_preds, dim=0)
        y_true = torch.cat(y_trues, dim=0)
        
        y_preds = [torch.zeros_like(y_pred) for _ in range(self.world_size)]
        y_trues = [torch.zeros_like(y_true) for _ in range(self.world_size)]
        dist.all_gather(y_preds, y_pred)
        dist.all_gather(y_trues, y_true)
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)

        acc = self.acc(torch.softmax(y_preds, dim=-1), y_trues.squeeze(-1)).cpu().item()
        f1 = self.f1(torch.softmax(y_preds, dim=-1), y_trues.squeeze(-1)).cpu().item()
        metrics = {'acc': acc, 'f1': f1}

        self.model.train()
        return metrics

    def train(self):
        _, train_loader = self._get_data(flag='train')

        path = self.logging_dir
        checkpoint_dir = os.path.join(path, 'checkpoint')
        if self.device == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            os.makedirs(os.path.join(self.logging_dir, 'graph'), exist_ok=True)

        args_file = os.path.join(self.logging_dir, 'args.json')
        if self.device == 0:
            with open(args_file, 'w') as f:
                json.dump(vars(self.args), f, indent=4, sort_keys=True)

        early_stopping = EarlyStopping(patience=self.args.patience, 
                                       verbose=True, if_max=True, device=self.device)
        
        model_optim = self._select_optimizer()
        scheduler = self._select_scheduler(model_optim)
        saver = CheckpointSaver(checkpoint_dir, metric_name='F1',
                                  maximize_metric=True)

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
                    y = y.to(self.device)

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

                        if self.args.use_fft:
                            x = torch.fft.rfft(x)[..., 1:]
                            x = torch.log(torch.abs(x) + 1e-8)

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
                        seq_len = torch.ones(x.shape[0], dtype=torch.int64).cuda() * self.args.input_len
                        y_pred = self.model(x, seq_len, supports)
                        loss = self.criterion(y_pred, y.squeeze(-1).long())
                    elif self.args.model in ['TimesNet']:
                        y_pred = self.model(x)
                        # loss = self.criterion(y_pred, y, reduction='mean')
                        loss = self.criterion(y_pred, y.squeeze(-1).long())
                    elif self.args.model in ['VQ_MTM', 'Ti_MAE']:
                        y_pred = self.model(x)
                        loss = self.criterion(y_pred, y.squeeze(-1).long())
                    else:
                        pass
                    
                    loss_val = loss.item()
                    acc = self.acc(torch.softmax(y_pred, dim=-1), y.squeeze(-1)).item()
                    loss.backward()

                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.max_norm)
                    self.steps += batch_size * self.world_size
                    model_optim.step()
                    
                    progress_bar.update(batch_size * self.world_size)
                    progress_bar.set_postfix(loss=loss_val, acc=acc)

                    self.logging.add_scalar('train/loss', loss_val, self.steps)
                    self.logging.add_scalar('train/lr', model_optim.param_groups[0]['lr'], self.steps)
                    self.logging.add_scalar('train/acc', acc, self.steps)

            if (epoch+1) % self.args.eval_every == 0:
                vali_metrics = self.vali()
                self.logging.add_scalar('vali/acc', vali_metrics['acc'], epoch)
                self.logging.add_scalar('vali/f1', vali_metrics['f1'], epoch)
                saver.save(epoch, self.model, model_optim, vali_metrics['f1'])
                early_stopping(vali_metrics['f1'])
                if early_stopping.early_stop:
                    break
                barrier()
            
            if self.args.use_scheduler:
                scheduler.step()
        return

    def test(self, model_file:str='best.pth.tar', model_dir:str=None):
        _, test_loader = self._get_data(flag='eval')
        if model_dir is None:
            path_dir = os.path.join(self.logging_dir, 'checkpoint')
        else:
            path_dir = model_dir
        path = os.path.join(path_dir, model_file)
        load_model_checkpoint(path, self.model, map_location=self.device, if_strict=True)

        self.model.eval()
        y_trues = []
        y_preds = []
        with torch.no_grad():
            for x, y, _ in tqdm(test_loader, disable=(self.device != 0)):
                x = x.float().to(self.device)
                y = y.to(self.device)

                # get adjmat, supports
                if self.args.use_graph:
                    _, supports = get_supports(self.args, x)
                else:
                    supports = None

                if self.args.using_patch:
                    batch_size, node_num, seq_len = x.shape
                    x = x.reshape(batch_size, node_num, -1, self.args.freq)
                    x = x.permute(0, 2, 1, 3)

                if self.args.use_fft:
                    x = torch.fft.rfft(x)[..., 1:]
                    x = torch.log(torch.abs(x) + 1e-8)

                if self.args.model in ['DCRNN']:
                    seq_len = torch.ones(x.shape[0], dtype=torch.int64).cuda() * self.args.input_len
                    y_pred = self.model(x, seq_len, supports)
                elif self.args.model in ['TimesNet']:
                    y_pred = self.model(x)
                elif self.args.model in ['VQ_MTM', 'Ti_MAE']:
                    y_pred = self.model(x)
                else:
                    raise NotImplementedError

                y_preds.append(y_pred)
                y_trues.append(y)

        y_pred = torch.cat(y_preds, dim=0)
        y_true = torch.cat(y_trues, dim=0)

        y_preds = [torch.zeros_like(y_pred) for _ in range(self.world_size)]
        y_trues = [torch.zeros_like(y_true) for _ in range(self.world_size)]
        dist.all_gather(y_preds, y_pred)
        dist.all_gather(y_trues, y_true)
        y_preds = torch.cat(y_preds, dim=0)
        y_trues = torch.cat(y_trues, dim=0)

        acc = self.acc(torch.softmax(y_preds, dim=-1), y_trues.squeeze(-1).int()).cpu().item()
        f1 = self.f1(torch.softmax(y_preds, dim=-1), y_trues.squeeze(-1).int()).cpu().item()

        metric = MulticlassConfusionMatrix(num_classes=4).to(self.device)
        metric.update(y_preds, y_trues.squeeze(-1).int())
        fig_, ax_ = metric.plot()
        self.logging.add_figure('test/Confusion_Matrix', fig_)

        self.logging.add_scalar('test/acc', acc)
        self.logging.add_scalar('test/f1_score', f1)
        return
    
