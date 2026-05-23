import json
from typing import List, Optional, Tuple

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

_GT_SCORE_KEY = {
    'summe': 'user_summary',
    'tvsum': 'user_summary',
    'mrhisum': 'gt_summary',
}


class VideoRecord:
    __slots__ = (
        'video_name', 'features', 'gtscore',
        'user_summary', 'shot_bound', 'n_frames', 'positions',
    )

    def __init__(
        self,
        video_name: str,
        features: Tensor,
        gtscore: Tensor,
        user_summary: np.ndarray,
        shot_bound: np.ndarray,
        n_frames: int,
        positions: np.ndarray,
    ):
        self.video_name = video_name
        self.features = features
        self.gtscore = gtscore
        self.user_summary = user_summary
        self.shot_bound = shot_bound
        self.n_frames = n_frames
        self.positions = positions


class VideoData(Dataset):
    def __init__(self, mode: str, video_type: str, split_index: int):
        self.mode = mode
        self.name = video_type.lower()
        self.split_index = split_index
        self.records: List[VideoRecord] = []
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

    def _read_record(self, hdf: h5py.File, video_name: str, dataset_name: str) -> VideoRecord:
        features = torch.tensor(
            np.array(hdf[f'{video_name}/features']), dtype=torch.float32
        )
        gtscore = torch.tensor(
            np.array(hdf[f'{video_name}/gtscore']), dtype=torch.float32
        )
        gt_key = _GT_SCORE_KEY.get(dataset_name, 'user_summary')
        user_summary = np.array(hdf[f'{video_name}/{gt_key}'])
        shot_bound = np.array(hdf[f'{video_name}/change_points'])

        if dataset_name == 'mrhisum':
            n_frames = features.shape[0]
            positions = np.arange(n_frames, dtype=np.int32)
        else:
            n_frames = int(np.array(hdf[f'{video_name}/n_frames']))
            positions = np.array(hdf[f'{video_name}/picks'], dtype=np.int32)

        return VideoRecord(
            video_name=video_name,
            features=features,
            gtscore=gtscore,
            user_summary=user_summary,
            shot_bound=shot_bound,
            n_frames=n_frames,
            positions=positions,
        )

    def _load_single(self, name: str) -> None:
        filename = _DATASET_FILES[name]
        self.split = self._load_split(_SPLIT_FILES[name], name)
        with h5py.File(filename, 'r') as hdf:
            for video_name in self.split[f'{self.mode}_keys']:
                self.records.append(self._read_record(hdf, video_name, name))

    def _load_combined(self, dataset_names: List[str]) -> None:
        combined_keys: List[str] = []
        for name in dataset_names:
            split = self._load_split(_SPLIT_FILES[name], name)
            with h5py.File(_DATASET_FILES[name], 'r') as hdf:
                for video_name in split[f'{self.mode}_keys']:
                    self.records.append(self._read_record(hdf, video_name, name))
                    combined_keys.append(video_name)
        self.split = {f'{self.mode}_keys': combined_keys}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> VideoRecord:
        return self.records[index]


def get_loader(mode: str, video_type: str, split_index: int):
    dataset = VideoData(mode, video_type, split_index)
    if mode.lower() == 'train':
        return DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    return dataset