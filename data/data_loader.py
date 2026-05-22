import json
from typing import List, Tuple

import h5py
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from configs.constants import (
    SUMME_DATASET_PATH,
    TVSUM_DATASET_PATH,
    MRHISUM_DATASET_PATH,
    SUMME_SPLIT_FILE_PATH,
    TVSUM_SPLIT_FILE_PATH,
    MRHISUM_SPLIT_FILE_PATH,
)

_DATASET_FILES = {
    'summe': SUMME_DATASET_PATH,
    'tvsum': TVSUM_DATASET_PATH,
    'mrhisum': MRHISUM_DATASET_PATH,
}

_SPLIT_FILES = {
    'summe': SUMME_SPLIT_FILE_PATH,
    'tvsum': TVSUM_SPLIT_FILE_PATH,
    'mrhisum': MRHISUM_SPLIT_FILE_PATH,
}


class VideoData(Dataset):
    def __init__(self, mode: str, video_type: str, split_index: int):
        self.mode = mode
        self.name = video_type.lower()
        self.split_index = split_index
        self.list_frame_features: List[Tensor] = []
        self.list_gtscores: List[Tensor] = []
        self.split: dict = {}

        if self.name == 'both':
            self._load_combined(['summe', 'tvsum'])
        else:
            self._load_single(self.name)

    def _load_split(self, splits_filename: str, dataset_name: str) -> dict:
        with open(splits_filename) as f:
            data = json.load(f)
        if dataset_name == 'mrhisum':
            return data
        for i, split in enumerate(data):
            if i == self.split_index:
                return split
        raise IndexError(f"split_index {self.split_index} out of range in {splits_filename}")

    def _load_single(self, name: str) -> None:
        filename = _DATASET_FILES[name]
        splits_filename = _SPLIT_FILES[name]
        self.split = self._load_split(splits_filename, name)

        with h5py.File(filename, 'r') as hdf:
            for video_name in self.split[f'{self.mode}_keys']:
                self.list_frame_features.append(
                    torch.tensor(np.array(hdf[f'{video_name}/features']), dtype=torch.float32)
                )
                self.list_gtscores.append(
                    torch.tensor(np.array(hdf[f'{video_name}/gtscore']), dtype=torch.float32)
                )

    def _load_combined(self, dataset_names: List[str]) -> None:
        combined_split: dict = {f'{self.mode}_keys': []}
        for name in dataset_names:
            filename = _DATASET_FILES[name]
            split = self._load_split(_SPLIT_FILES[name], name)
            with h5py.File(filename, 'r') as hdf:
                for video_name in split[f'{self.mode}_keys']:
                    self.list_frame_features.append(
                        torch.tensor(np.array(hdf[f'{video_name}/features']), dtype=torch.float32)
                    )
                    self.list_gtscores.append(
                        torch.tensor(np.array(hdf[f'{video_name}/gtscore']), dtype=torch.float32)
                    )
                    combined_split[f'{self.mode}_keys'].append(video_name)
        self.split = combined_split

    def __len__(self) -> int:
        return len(self.split[f'{self.mode}_keys'])

    def __getitem__(self, index: int) -> Tuple[Tensor, object]:
        video_name = self.split[f'{self.mode}_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore = self.list_gtscores[index]
        if self.mode == 'test':
            return frame_features, video_name
        return frame_features, gtscore


def get_loader(mode: str, video_type: str, split_index: int):
    dataset = VideoData(mode, video_type, split_index)
    if mode.lower() == 'train':
        return DataLoader(dataset, batch_size=1, shuffle=True)
    return dataset