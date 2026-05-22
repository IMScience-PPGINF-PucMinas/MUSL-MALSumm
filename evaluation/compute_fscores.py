import argparse
import json
from os import listdir
from typing import List

import h5py
import numpy as np

from .evaluation_metrics import evaluate_summary
from inference.generate_summary import generate_summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='../Summaries/SumMe/results/split0')
    parser.add_argument('--dataset', type=str, default='SumMe')
    parser.add_argument('--eval', type=str, default='max', choices=['max', 'avg'])
    return parser.parse_args()


def compute_fscores(path: str, dataset: str, eval_method: str) -> None:
    dataset_path = (
        f'../data/datasets/{dataset}/eccv16_dataset_{dataset.lower()}_google_pool5.h5'
    )

    result_files = sorted(
        [f for f in listdir(path) if f.endswith('.json')],
        key=lambda x: int(x[6:-5]),
    )

    f_score_epochs: List[float] = []

    for epoch_file in result_files:
        with open(f'{path}/{epoch_file}') as f:
            data = json.load(f)
        keys = list(data.keys())
        all_scores = [np.asarray(data[k]) for k in keys]

        all_user_summary: List[np.ndarray] = []
        all_shot_bound: List[np.ndarray] = []
        all_nframes: List[int] = []
        all_positions: List[np.ndarray] = []

        with h5py.File(dataset_path, 'r') as hdf:
            for video_name in keys:
                video_index = video_name[6:]
                all_user_summary.append(np.array(hdf[f'video_{video_index}/user_summary']))
                all_shot_bound.append(np.array(hdf[f'video_{video_index}/change_points']))
                all_nframes.append(int(np.array(hdf[f'video_{video_index}/n_frames'])))
                all_positions.append(np.array(hdf[f'video_{video_index}/picks']))

        all_summaries = generate_summary(all_shot_bound, all_scores, all_nframes, all_positions)

        epoch_f_scores = [
            evaluate_summary(all_summaries[i], all_user_summary[i], eval_method)
            for i in range(len(all_summaries))
        ]

        mean_f = float(np.mean(epoch_f_scores))
        f_score_epochs.append(mean_f)
        print(f'f_score: {mean_f:.4f}')

    with open(f'{path}/f_scores.txt', 'w') as f:
        json.dump(f_score_epochs, f)


if __name__ == '__main__':
    args = _parse_args()
    compute_fscores(args.path, args.dataset, args.eval)