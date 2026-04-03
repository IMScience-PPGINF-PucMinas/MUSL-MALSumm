import argparse
import pprint
from pathlib import Path
import torch
from utils.utils import get_paths


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:
    def __init__(self, **kwargs):
        self.log_dir   = None
        self.score_dir = None
        self.save_dir  = None
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        paths = get_paths(video_type)
        self.log_dir   = f"{paths['log_dir']}/split{self.split_index}"
        self.score_dir = f"{paths['score_dir']}/split{self.split_index}"
        self.save_dir  = f"{paths['save_dir']}/split{self.split_index}"

    def __repr__(self):
        return 'Configurations\n' + pprint.pformat(self.__dict__)


def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=str2bool, default=False)
    parser.add_argument('--video_type', type=str, default='SumMe')
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--fusion', type=str, default='add')
    parser.add_argument('--n_segments', type=int, default=4)
    parser.add_argument('--pos_enc', type=str, default='absolute')
    parser.add_argument('--heads', type=int, default=8)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--clip', type=float, default=5.0)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--l2_req', type=float, default=1e-5)
    parser.add_argument('--split_index', type=int, default=0)
    parser.add_argument('--init_type', type=str, default='xavier')
    parser.add_argument('--init_gain', type=float, default=None)
    parser.add_argument('--early_stop_patience', type=int, default=30)
    parser.add_argument('--lr_scheduler_patience', type=int, default=10)
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5)
    parser.add_argument('--lambda_rank', type=float, default=0.1)
    parser.add_argument('--alpha_rl', type=float, default=0.1)
    parser.add_argument('--alpha_rl_end', type=float, default=0.01)
    parser.add_argument('--use_aug_splits', type=str2bool, default=True)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--max_seq_len', type=int, default=500)

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)
    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    print(config)
