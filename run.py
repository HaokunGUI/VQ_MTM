import argparse
import torch
from exp.exp_anomaly_detection import Exp_Anomaly_Detection
from exp.exp_ssl import Exp_SSL
from exp.exp_classification import Exp_Classification
import torch.multiprocessing
from utils.tools import ddp_setup, ddp_cleanup, seed_torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def main(args: argparse.Namespace):
    seed_torch(args.seed)
    if args.use_gpu:
        ddp_setup()
        rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0

    if args.task_name == 'anomaly_detection':
        Exp = Exp_Anomaly_Detection
    elif args.task_name == 'ssl':
        Exp = Exp_SSL
    elif args.task_name == 'classification':
        Exp = Exp_Classification
    else:
        raise ValueError('task name must be in [anomaly_detection, classification, ssl]')

    exp = Exp(args)
    if rank == 0:
        print('>>>>>>> training : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.train()
    if rank == 0:
        print('>>>>>>> testing : <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
    exp.test()
    torch.cuda.empty_cache()
    
    if args.use_gpu:
        ddp_cleanup()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='ssl',
                        help='task name, options:[classification, anomaly_detection, ssl]')
    parser.add_argument('--model', type=str, required=True, default='DCRNN',
                        help='model name, options: [DCRNN, TimesNet]')
    parser.add_argument('--log_dir', type=str, default='/home/guihaokun/Time-Series-Pretrain/logging', help='log dir')
    parser.add_argument('--seed', type=int, default=1029, help='random seed')
    parser.add_argument('--last_train_path', type=str, default=None, help='last train model path')
    parser.add_argument('--activation', type=str, default='gelu', help='activation function, options:[relu, gelu]')

    # data loader
    parser.add_argument('--dataset', type=str, default='TUSZ', help='dataset type, options:[TUSZ]')
    parser.add_argument('--root_path', type=str, default='/mnt/data/guihaokun/resample/tuh_eeg_seizure/', help='root path of the data file')
    parser.add_argument('--classification_dir', type=str, default='/mnt/data/guihaokun/classification/', help='classification dir')
    parser.add_argument('--marker_dir', type=str, default='/home/guihaokun/Time-Series-Pretrain/data', help='marker dir')
    parser.add_argument('--data_augment', action='store_true', help='use data augment or not', default=False)
    parser.add_argument('--normalize', action='store_true', help='normalize data or not', default=False)
    parser.add_argument('--train_batch_size', type=int, default=64, help='batch size of train input data')
    parser.add_argument('--test_batch_size', type=int, default=64, help='batch size of test input data')
    parser.add_argument('--num_workers', type=int, default=16, help='data loader num workers')
    parser.add_argument('--freq', type=int, default=250, help='sample frequency')

    # ssl task
    parser.add_argument('--input_len', type=int, default=60, help='input sequence length')
    parser.add_argument('--output_len', type=int, default=12, help='prediction sequence length')
    parser.add_argument('--time_step_len', type=int, default=1, help='time step length')
    parser.add_argument('--use_fft', action='store_true', help='use fft or not', default=False)
    parser.add_argument('--loss_fn', type=str, default='mae', help='loss function, options:[mse, mae]')

    # classification task
    parser.add_argument('--num_classes', type=int, default=4, help='number of classes')

    # detection task
    parser.add_argument('--scale_ratio', type=float, default=1.0, help='scale ratio of train data')
    parser.add_argument('--balanced', action='store_true', help='balanced data or not', default=False)

    # graph setting
    parser.add_argument('--graph_type', type=str, default='correlation', help='graph type, option:[distance, correlation]')
    parser.add_argument('--top_k', type=int, default=3, help='top k in graph or top k in TimesNet')
    parser.add_argument('--directed', action='store_true', help='directed graph or not', default=False)
    parser.add_argument('--filter_type', type=str, default='dual_random_walk', help='filter type')

    # model define
    parser.add_argument('--num_nodes',type=int, default=19, help='Number of nodes in graph.')
    parser.add_argument('--num_rnn_layers', type=int, default=2, help='Number of RNN layers in encoder and/or decoder.')
    parser.add_argument('--rnn_units', type=int, default=64, help='Number of hidden units in DCRNN.')
    parser.add_argument('--dcgru_activation', type=str, choices=('relu', 'tanh'), default='tanh', help='Nonlinear activation used in DCGRU cells.')
    parser.add_argument('--input_dim', type=int, default=None, help='Input seq feature dim.')
    parser.add_argument('--output_dim', type=int, default=None, help='Output seq feature dim.')
    parser.add_argument('--max_diffusion_step', type=int, default=2, help='Maximum diffusion step.')
    parser.add_argument('--cl_decay_steps', type=int, default=3000, help='Scheduled sampling decay steps.')
    parser.add_argument('--use_curriculum_learning', default=False, action='store_true', help='Whether to use curriculum training for seq-seq model.')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability.')
    parser.add_argument('--d_hidden', type=int, default=8, help='Hidden state dimension.')
    parser.add_argument('--num_kernels', type=int, default=5, help='Number of each kind of kernel.')
    parser.add_argument('--d_model', type=int, default=16, help='hidden dimension of channels')
    parser.add_argument('--e_layers', type=int, default=3, help='Number of encoder layers.')
    parser.add_argument('--attn_head', type=int, default=4, help='Number of attention heads.')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size.')
    parser.add_argument('--enc_type', type=str, default='rel', help='Encoder type, options:[abs, rel]')
    parser.add_argument('--linear_dropout', type=float, default=0.5, help='linear dropout ratio')
    parser.add_argument('--hidden_channels', type=int, default=16, help='hidden channels of conv')
    parser.add_argument('--d_layers', type=int, default=3, help='Number of decoder layers.')
    parser.add_argument('--global_pool', action='store_true', default=False, help='global pool or not')

    # quantization
    parser.add_argument('--codebook_item', type=int, default=512, help='number of embedding vectors')
    parser.add_argument('--codebook_num', type=int, default=4, help='number of codebooks')

    # masking
    parser.add_argument('--mask_ratio', type=float, default=0.2, help='mask ratio')
    parser.add_argument('--mask_length', type=int, default=10, help='mask length')
    parser.add_argument('--no_overlap', action='store_true', default=False, help='mask overlap or not')
    parser.add_argument('--min_space', type=int, default=1, help='min space between mask')
    parser.add_argument('--mask_dropout', type=float, default=0.0, help='mask dropout ratio')
    parser.add_argument('--mask_type', type=str, default='poisson', help='mask type, options:[static, uniform, normal, poisson]')
    parser.add_argument('--lm', type=int, default=3, help='lm of poisson mask')

    # optimization
    parser.add_argument('--num_epochs', type=int, default=60, help='train epochs')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=5e-4, help='optimizer learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='optimizer weight decay')
    parser.add_argument('--max_norm', type=float, default=1.0, help='max norm of grad')
    parser.add_argument('--use_scheduler', action='store_true', default=False, help='use scheduler or not')
    parser.add_argument('--warmup_epochs', type=int, default=20, help='warmup epochs')

    # SimMTM
    parser.add_argument('--temperature', type=float, default=0.1, help='temperature of softmax')
    parser.add_argument('--positive_nums', type=int, default=2, help='positive nums of contrastive learning')
    parser.add_argument('--dimension', type=int, default=64, help='dimension of SimMTM')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')

    # log setting
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate every X epochs')
    parser.add_argument('--adj_every', type=int, default=10, help='display adj matrix every X epochs')

    # pretrain
    parser.add_argument('--pretrained_path', type=str, default=None, help='pretrain model path')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu:
        args.gpu = 0
    
    if args.model in ['DCRNN']:
        args.use_graph = True
    else:
        args.use_graph = False

    if args.model in ['DCRNN']:
        args.using_patch = True
    else:
        args.using_patch = False

    if args.using_patch:
        args.input_dim = args.freq
        args.output_dim = args.freq
    else:
        args.input_dim = args.freq * args.input_len
        args.output_dim = args.freq * args.output_len
    
    if args.use_fft:
        args.input_dim = args.input_dim // 2
        args.output_dim = args.output_dim // 2
    
    if args.pretrained_path is not None:
        args.use_pretrained = True
    else:
        args.use_pretrained = False

    if args.last_train_path is not None:
        args.continue_train = True
    else:
        args.continue_train = False

    main(args)