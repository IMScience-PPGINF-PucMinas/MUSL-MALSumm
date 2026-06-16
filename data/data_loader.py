import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from configs.constants import *


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        self.mode = mode
        self.name = video_type.lower()
        self.split_index = split_index

        self.dataset_paths = {
            'summe':   SUMME_DATASET_PATH,
            'tvsum':   TVSUM_DATASET_PATH,
            'mrhisum': MRHISUM_DATASET_PATH,
        }
        self.splits_filename = 'data/splits/' + self.name + '_splits.json'

        self.list_frame_features, self.list_gtscores = [], []

        if self.name == 'both':
            self._load_both(mode, split_index)
        else:
            self._load_single(mode, split_index)

    def _load_both(self, mode, split_index):
        """Load SumMe + TVSum together."""
        splits_filenames = {
            'summe': SUMME_SPLIT_FILE_PATH,
            'tvsum': TVSUM_SPLIT_FILE_PATH,
        }
        for vtype in ('summe', 'tvsum'):
            filename = self.dataset_paths[vtype]  # FIX: dict lookup, sem TypeError
            with open(splits_filenames[vtype]) as f:
                data = json.load(f)
            for i, split in enumerate(data):
                if i == split_index:
                    self.split = split
                    break

            with h5py.File(filename, 'r') as hdf:
                for video_name in self.split[mode + '_keys']:
                    frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                    gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                    print(f"Loaded video {video_name}: features {frame_features.shape}, gtscore {gtscore.shape}")
                    self.list_frame_features.append(frame_features)
                    self.list_gtscores.append(gtscore)

    def _load_single(self, mode, split_index):
        """Load a single dataset (summe / tvsum / mrhisum)."""
        filename = self.dataset_paths[self.name]

        if self.name == 'mrhisum':
            with open(self.splits_filename) as f:
                self.split = json.load(f)
        else:
            with open(self.splits_filename) as f:
                data = json.load(f)
            for i, split in enumerate(data):
                if i == split_index:
                    self.split = split
                    break

        with h5py.File(filename, 'r') as hdf:
            for video_name in self.split[mode + '_keys']:
                frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
                gtscore = torch.Tensor(np.array(hdf[video_name + '/gtscore']))
                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)

    def __len__(self):
        """ Function to be called for the `len` operator of `VideoData` Dataset. """
        return len(self.split[self.mode + '_keys'])

    def __getitem__(self, index):

        video_name = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]

        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features, gtscore


def _worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    import random
    random.seed(worker_seed)


def get_loader(mode, video_type, split_index, seed=None):
    vd = VideoData(mode, video_type, split_index)

    if mode.lower() == 'train':
        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
        return DataLoader(
            vd,
            batch_size=1,
            shuffle=True,
            worker_init_fn=_worker_init_fn,
            generator=generator,
        )
    else:
        return vd


if __name__ == '__main__':
    pass