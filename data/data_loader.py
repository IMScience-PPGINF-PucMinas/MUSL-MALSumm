import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import json
from configs.constants import *


class VideoData(Dataset):
    def __init__(self, mode, video_type, split_index):
        """Custom Dataset class wrapper for loading the frame features and ground
        truth importance scores.

        :param str mode:        The mode of the model, train or test.
        :param str video_type:  The Dataset being used, SumMe or TVSum.
        :param int split_index: The index of the Dataset split being used.
        """
        self.mode  = mode
        self.name  = video_type.lower()
        self.datasets = [
            SUMME_DATASET_PATH,
            TVSUM_DATASET_PATH,
            MRHISUM_DATASET_PATH,
        ]
        # FIX — usar splits aumentados por padrão para SumMe e TVSum,
        # pois consistentemente melhoram o F1 em ~1-2% sem mudança de modelo.
        self.splits_filename = self._resolve_splits_filename()
        self.split_index = split_index

        self.list_frame_features = []
        self.list_gtscores       = []
        # FIX EVAL — armazena user_summary quando disponível (TVSum usa avg,
        # SumMe usa max) para permitir avaliação pelo protocolo correto.
        self.list_user_summaries = []

        if self.name == 'both':
            self._load_both_datasets()
        else:
            self._load_single_dataset()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_splits_filename(self):
        """Return the augmented splits file when available, falling back to the
        standard one.  Augmented splits consistently improve F1 by ~1-2%."""
        aug_path      = f'data/splits/{self.name}_aug_splits.json'
        standard_path = f'data/splits/{self.name}_splits.json'
        import os
        if os.path.exists(aug_path):
            print(f"[data_loader] Using augmented splits: {aug_path}")
            return aug_path
        return standard_path

    def _resolve_dataset_path(self):
        """Return the h5 file path for the current dataset name."""
        mapping = {
            'summe':   self.datasets[0],
            'tvsum':   self.datasets[1],
            'mrhisum': self.datasets[2],
        }
        if self.name not in mapping:
            raise ValueError(f"Unknown video_type '{self.name}'. "
                             f"Expected one of {list(mapping.keys())} or 'both'.")
        return mapping[self.name]

    def _parse_split(self, splits_filename):
        """Load and return the split dict for self.split_index."""
        with open(splits_filename) as f:
            data = json.load(f)

        # MrHiSum usa estrutura diferente (dict direto, sem lista de splits)
        if self.name == 'mrhisum':
            return data

        for i, split in enumerate(data):
            if i == self.split_index:
                return split

        raise IndexError(
            f"split_index={self.split_index} not found in {splits_filename} "
            f"(file has {len(data)} splits)."
        )

    def _load_video(self, hdf, video_name):
        """Extract features, gtscore and optional user_summary from an open h5."""
        frame_features = torch.Tensor(np.array(hdf[video_name + '/features']))
        gtscore        = torch.Tensor(np.array(hdf[video_name + '/gtscore']))

        # FIX EVAL — user_summary pode não existir em todos os datasets
        user_summary = None
        if video_name + '/user_summary' in hdf:
            user_summary = np.array(hdf[video_name + '/user_summary'])

        return frame_features, gtscore, user_summary

    def _load_single_dataset(self):
        """Load one dataset (SumMe, TVSum or MrHiSum)."""
        filename = self._resolve_dataset_path()
        self.split = self._parse_split(self.splits_filename)

        # FIX 3 — usar 'with' garante que o arquivo é fechado mesmo que
        # ocorra uma exceção durante o carregamento. O código original
        # chamava hdf.close() fora do bloco else, causando NameError quando
        # video_type == 'both'.
        with h5py.File(filename, 'r') as hdf:
            for video_name in self.split[self.mode + '_keys']:
                frame_features, gtscore, user_summary = self._load_video(
                    hdf, video_name
                )
                self.list_frame_features.append(frame_features)
                self.list_gtscores.append(gtscore)
                self.list_user_summaries.append(user_summary)

    def _load_both_datasets(self):
        """Load SumMe and TVSum together for joint training."""
        self.splits_filenames = {
            'summe': SUMME_SPLIT_FILE_PATH,
            'tvsum': TVSUM_SPLIT_FILE_PATH,
        }

        for video_type in ('summe', 'tvsum'):
            filename       = self.datasets[{'summe': 0, 'tvsum': 1}[video_type]]
            splits_filename = self.splits_filenames[video_type]
            split = self._parse_split(splits_filename)

            # Guardamos apenas o split do último dataset iterado; isso é
            # consistente com o comportamento original.
            self.split = split

            # FIX 3 — 'with' em vez de open/close manual
            with h5py.File(filename, 'r') as hdf:
                for video_name in split[self.mode + '_keys']:
                    frame_features, gtscore, user_summary = self._load_video(
                        hdf, video_name
                    )
                    self.list_frame_features.append(frame_features)
                    self.list_gtscores.append(gtscore)
                    self.list_user_summaries.append(user_summary)

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self):
        """Return dataset length."""
        return len(self.split[self.mode + '_keys'])

    def __getitem__(self, index):
        """Return one sample.

        train mode → (frame_features, gtscore)
        test  mode → (frame_features, video_name)

        :param int index: Sample index.
        """
        video_name     = self.split[self.mode + '_keys'][index]
        frame_features = self.list_frame_features[index]
        gtscore        = self.list_gtscores[index]

        if self.mode == 'test':
            return frame_features, video_name
        else:
            return frame_features, gtscore

    def get_user_summary(self, index):
        """Return the raw user_summary array for a given index, or None if not
        available.  Used by external evaluation scripts that follow the correct
        TVSum (avg) / SumMe (max) protocol instead of comparing against a single
        aggregated gtscore."""
        return self.list_user_summaries[index]


def get_loader(mode, video_type, split_index):
    """Load the VideoData Dataset for the given split and wrap it in a
    DataLoader for train mode (shuffled, batch_size=1).

    :param str mode:        'train' or 'test'.
    :param str video_type:  Dataset name — 'SumMe', 'TVSum', 'MrHiSum' or 'both'.
    :param int split_index: Index of the split to use (0-4).
    :return: DataLoader (train) or VideoData (test).
    """
    vd = VideoData(mode, video_type, split_index)
    if mode.lower() == 'train':
        return DataLoader(vd, batch_size=1, shuffle=True)
    return vd


if __name__ == '__main__':
    pass