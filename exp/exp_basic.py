import os
import torch
from models import TimesNet, DCRNN, VQ_MTM, SimMTM, Ti_MAE, BIOT
from tensorboardX import SummaryWriter
from utils.tools import WriterFilter
import datetime


class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'TimesNet': TimesNet,
            'DCRNN': DCRNN,
            'VQ_MTM': VQ_MTM,
            'SimMTM': SimMTM,
            'Ti_MAE': Ti_MAE,
            'BIOT': BIOT,
        }
        self.device = self._acquire_device()
        self.model = self._build_model()
        self.scalar = self._get_scalar()
        self.world_size = int(os.environ["WORLD_SIZE"])
        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M")
        self.logging_dir = os.path.join(self.args.log_dir, self.args.task_name, self.args.model, f'{self.args.model}_{suffix}')
        log_dir = os.path.join(self.logging_dir, 'log')
        os.makedirs(log_dir, exist_ok=True)
        self.logging = WriterFilter(SummaryWriter(log_dir))
        self.criterion = self._select_criterion()

    def _select_criterion(self):
        raise NotImplementedError

    def _build_model(self):
        raise NotImplementedError

    def _acquire_device(self):
        if self.args.use_gpu:
            device = int(os.environ["LOCAL_RANK"])
        else:
            device = torch.device('cpu')
        return device
    
    def _get_scalar(self):
        raise NotImplementedError

    def _get_data(self):
        pass

    def vali(self, vali_loader, criterion):
        pass

    def train(self):
        pass

    def test(self, model_file:str='best.pth.tar'):
        pass
