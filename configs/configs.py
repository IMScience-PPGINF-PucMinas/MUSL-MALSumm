import argparse
import pprint
from pathlib import Path

import torch

from utils.utils import get_paths


def _str2bool(v: str) -> bool:
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:
    def __init__(self, **kwargs):
        self.log_dir = None
        self.score_dir = None
        self.save_dir = None
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._set_dataset_dirs(self.video_type)

    def _set_dataset_dirs(self, video_type: str) -> None:
        paths = get_paths(video_type)
        self.log_dir = f"{paths['log_dir']}/split{self.split_index}"
        self.score_dir = f"{paths['score_dir']}/split{self.split_index}"
        self.save_dir = f"{paths['save_dir']}/split{self.split_index}"

    def __repr__(self) -> str:
        return 'Configurations\n' + pprint.pformat(self.__dict__)


def get_config(parse: bool = True, **optional_kwargs) -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--verbose', type=_str2bool, default=False)
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

    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    kwargs = vars(parser.parse_args() if parse else parser.parse_known_args()[0])
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    print(get_config())