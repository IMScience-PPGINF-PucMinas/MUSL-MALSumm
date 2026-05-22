import argparse
import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
from matplotlib.patches import Patch

from evaluation.evaluation_metrics import evaluate_summary
from inference.generate_summary import generate_summary
from model.layers.summarizer import xLSTM
from utils.utils import get_paths

PALETTE = {
    'raw': '#4C72B0',
    'knapsack': '#DD8452',
    'gt': '#55A868',
    'shot_edge': '#CCCCCC',
    'bg_select': '#FFF3E0',
}


def _min_max_normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi - lo < eps:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def _sanitise_filename(name: str) -> str:
    name = str(name).strip().replace(' ', '_')
    name = re.sub(r'[^\w\-.]', '', name)
    return name or 'video'


def _build_shot_signals(
    scores_norm: np.ndarray,
    positions: np.ndarray,
    n_frames: int,
    shot_bound: np.ndarray,
    user_summary: np.ndarray,
    eval_metric: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int], List[int], float]:
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        frame_scores[pos[i]:pos[i + 1]] = scores_norm[i] if i < len(scores_norm) else 0.0

    shot_lengths = [int(s[1] - s[0] + 1) for s in shot_bound]
    shot_model_scores = np.array(
        [frame_scores[s[0]:s[1] + 1].mean() for s in shot_bound], dtype=np.float32
    )

    summary = generate_summary(
        [shot_bound], [scores_norm.tolist()], [n_frames], [positions]
    )[0]

    knapsack_selected = np.zeros(len(shot_bound), dtype=np.float32)
    selected_indices: List[int] = []
    for i, s in enumerate(shot_bound):
        if summary[s[0]:s[1] + 1].any():
            knapsack_selected[i] = 1.0
            selected_indices.append(i)

    f_score = evaluate_summary(summary, user_summary, eval_metric)

    gt_frame = np.atleast_2d(user_summary).mean(axis=0).astype(np.float32)
    shot_gt_scores = np.array(
        [gt_frame[s[0]:min(s[1] + 1, len(gt_frame))].mean() for s in shot_bound],
        dtype=np.float32,
    )
    shot_gt_scores = _min_max_normalize(shot_gt_scores)

    return shot_model_scores, knapsack_selected, shot_gt_scores, shot_lengths, selected_indices, f_score


def plot_video(
    video_id: str,
    shot_model_scores: np.ndarray,
    knapsack_selected: np.ndarray,
    shot_gt_scores: np.ndarray,
    shot_lengths: List[int],
    selected_indices: List[int],
    shot_bound: np.ndarray,
    video_name: str,
    dataset: str,
    output_dir: str,
    f_score: Optional[float] = None,
) -> str:
    n_shots = len(shot_model_scores)
    total_frames = sum(shot_lengths)

    shot_starts = np.array([shot_bound[i][0] for i in range(n_shots)], dtype=float)
    shot_ends = np.array([shot_bound[i][1] + 1 for i in range(n_shots)], dtype=float)
    shot_centers = (shot_starts + shot_ends) / 2.0

    fig, ax = plt.subplots(figsize=(14, 4.5), dpi=130)
    fig.patch.set_facecolor('#FAFAFA')
    ax.set_facecolor('#F5F5F5')

    for idx in selected_indices:
        ax.axvspan(shot_starts[idx], shot_ends[idx],
                   facecolor=PALETTE['bg_select'], alpha=0.50, zorder=1, linewidth=0)

    for i in range(n_shots - 1):
        ax.axvline(shot_ends[i], color=PALETTE['shot_edge'], linewidth=0.5, zorder=2, alpha=0.7)

    ax.step(shot_starts, shot_gt_scores, color=PALETTE['gt'], linewidth=1.4,
            where='post', alpha=0.85, zorder=3, label='Ground truth (mean, per shot)')
    ax.step(shot_starts, shot_model_scores, color=PALETTE['raw'], linewidth=1.4,
            where='post', alpha=0.90, zorder=4, label='Model score (normalised, per shot)')
    ax.step(shot_starts, knapsack_selected, color=PALETTE['knapsack'], linewidth=2.0,
            where='post', alpha=0.95, zorder=5, label='Knapsack selection (1 = selected)')

    if selected_indices:
        sel_x = shot_centers[selected_indices]
        sel_y = shot_model_scores[selected_indices]
        ax.scatter(sel_x, sel_y, color=PALETTE['knapsack'], s=28, zorder=6, alpha=0.85)

    ax.set_xlim(0, total_frames)
    ax.set_ylim(-0.05, 1.15)
    ax.set_xlabel('Frame position (shot boundaries)', fontsize=10, labelpad=6)
    ax.set_ylabel('Score / Selection', fontsize=10, labelpad=6)

    step = max(1, n_shots // 8)
    tick_positions = shot_starts[::step]
    tick_labels = [str(i * step) for i in range(len(tick_positions))]
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(shot_starts))
    ax.tick_params(axis='x', which='minor', length=2, color=PALETTE['shot_edge'])

    ax.text(0.5, -0.13, f'{n_shots} shots  ·  {total_frames:,} frames total',
            transform=ax.transAxes, ha='center', fontsize=8, color='#777777')

    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
    ax.tick_params(axis='both', labelsize=8.5, length=3)
    ax.grid(axis='y', linestyle='--', linewidth=0.5, alpha=0.5, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE['raw'], linewidth=1.5,
                       label='Model score (normalised, per shot)'),
            plt.Line2D([0], [0], color=PALETTE['knapsack'], linewidth=2.0,
                       label='Knapsack selection (1 = selected)'),
            plt.Line2D([0], [0], color=PALETTE['gt'], linewidth=1.5,
                       label='Ground truth (mean, per shot)'),
            Patch(facecolor=PALETTE['bg_select'], edgecolor='none', alpha=0.7,
                  label='Selected region'),
        ],
        loc='upper right', fontsize=8, framealpha=0.85,
        edgecolor='#CCCCCC', handlelength=2.0,
    )

    title_parts = [f'{dataset}  ·  {video_name}  ({video_id})']
    if f_score is not None:
        title_parts.append(f'F1 = {f_score:.2f}%')
    ax.set_title('   |   '.join(title_parts), fontsize=10.5, fontweight='bold', pad=8, loc='left')

    pct_shots = len(selected_indices) / n_shots * 100
    pct_frames = sum(shot_lengths[i] for i in selected_indices) / total_frames * 100
    ax.text(
        0.995, 1.02,
        f'Selected: {len(selected_indices)}/{n_shots} shots ({pct_shots:.0f}%)  ·  {pct_frames:.1f}% of frames',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='#555555',
    )

    out_folder = os.path.join(output_dir, dataset)
    os.makedirs(out_folder, exist_ok=True)
    safe_name = _sanitise_filename(video_name or video_id)
    out_path = os.path.join(out_folder, f'{safe_name}.png')

    plt.tight_layout(pad=1.4)
    plt.savefig(out_path, bbox_inches='tight')
    plt.close(fig)

    return out_path


def plot_split(
    split_id: int,
    dataset: str,
    model_path: str,
    epoch_fname: str,
    dataset_path: str,
    split_data,
    model_kwargs: dict,
    output_dir: str,
    verbose: bool = False,
) -> List[str]:
    eval_metric = 'avg' if dataset.lower() == 'tvsum' else 'max'
    test_keys = (
        split_data[split_id]['test_keys']
        if isinstance(split_data, list)
        else split_data['test_keys']
    )

    model = xLSTM(**model_kwargs)
    model.load_state_dict(torch.load(os.path.join(model_path, epoch_fname), map_location='cpu'))
    model.eval()

    saved_paths: List[str] = []

    with h5py.File(dataset_path, 'r') as hdf:
        for video_id in test_keys:
            if dataset.lower() == 'summe':
                try:
                    if int(video_id.split('_')[1]) > 25:
                        continue
                except (IndexError, ValueError):
                    pass

            features = torch.tensor(
                np.array(hdf[f'{video_id}/features']), dtype=torch.float32
            ).view(-1, 1024)
            shot_bound = np.array(hdf[f'{video_id}/change_points'])
            n_frames = int(np.array(hdf[f'{video_id}/n_frames']))
            positions = np.array(hdf[f'{video_id}/picks'])

            if f'{video_id}/user_summary' in hdf:
                user_summary = np.array(hdf[f'{video_id}/user_summary'])
            elif f'{video_id}/gt_summary' in hdf:
                user_summary = np.array(hdf[f'{video_id}/gt_summary'])
            else:
                logging.warning(f'No ground truth found for {video_id} — skipping')
                continue

            video_name = video_id
            if f'{video_id}/video_name' in hdf:
                video_name = str(
                    np.array(hdf[f'{video_id}/video_name']).astype(str, copy=False)
                )

            with torch.no_grad():
                scores, _ = model(features)
                scores = scores.squeeze(0).cpu().numpy()

            scores_norm = _min_max_normalize(scores)

            (shot_model_scores, knapsack_selected,
             shot_gt_scores, shot_lengths,
             selected_indices, f_score) = _build_shot_signals(
                scores_norm, positions, n_frames, shot_bound, user_summary, eval_metric
            )

            out_path = plot_video(
                video_id=video_id,
                shot_model_scores=shot_model_scores,
                knapsack_selected=knapsack_selected,
                shot_gt_scores=shot_gt_scores,
                shot_lengths=shot_lengths,
                selected_indices=selected_indices,
                shot_bound=shot_bound,
                video_name=video_name,
                dataset=dataset,
                output_dir=output_dir,
                f_score=f_score,
            )
            saved_paths.append(out_path)

            msg = f'  {video_id} ({video_name}) → {out_path}  F1={f_score:.2f}%'
            if verbose:
                logging.info(msg)
            else:
                print(msg)

    return saved_paths


def _resolve_epoch_fname(model_path: str, epoch_arg: str) -> Tuple[str, int]:
    if str(epoch_arg).lower() == 'best':
        fscores_path = os.path.join(model_path, 'f_scores.txt')
        if not os.path.exists(fscores_path):
            raise FileNotFoundError(
                f'f_scores.txt not found in {model_path}. '
                'Run compute_fscores.py first, or pass --epoch <N>.'
            )
        with open(fscores_path) as fp:
            content = fp.read().strip()
        try:
            scores = json.loads(content)
        except json.JSONDecodeError:
            scores = [float(x) for x in content.splitlines()]
        best = int(np.argmax(scores))
        return f'epoch-{best}.pkl', best

    epoch_num = int(epoch_arg)
    return f'epoch-{epoch_num}.pkl', epoch_num


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Generate per-video score charts for a trained xLSTM checkpoint.'
    )
    parser.add_argument('--dataset', type=str, default='SumMe')
    parser.add_argument('--model_version', type=str, default='')
    parser.add_argument('--split', type=int, default=0)
    parser.add_argument('--all_splits', action='store_true')
    parser.add_argument('--epoch', type=str, default='best')
    parser.add_argument('--output_dir', type=str, default='plots')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)

    args = vars(parser.parse_args())

    dataset = args['dataset']
    model_version = args['model_version']
    output_dir = args['output_dir']
    verbose = args['verbose']
    split_ids = list(range(5)) if args['all_splits'] else [args['split']]

    model_kwargs = dict(
        input_size=1024,
        output_size=1024,
        num_segments=4,
        hidden_dim=args['hidden_dim'],
        num_layers=args['num_layers'],
        dropout=args['dropout'],
    )

    paths = get_paths(dataset)
    dataset_path = paths['dataset']
    split_file = paths['split']

    with open(split_file) as fp:
        split_data = json.load(fp)

    total_saved: List[str] = []

    for split_id in split_ids:
        model_path = f'Summaries/xLSTM/{dataset}{model_version}/models/split{split_id}'
        try:
            epoch_fname, epoch_num = _resolve_epoch_fname(model_path, args['epoch'])
        except FileNotFoundError as e:
            logging.error(str(e))
            continue

        print(f'\nSplit {split_id} — epoch {epoch_num} — generating charts for {dataset}...')

        saved = plot_split(
            split_id=split_id,
            dataset=dataset,
            model_path=model_path,
            epoch_fname=epoch_fname,
            dataset_path=dataset_path,
            split_data=split_data,
            model_kwargs=model_kwargs,
            output_dir=output_dir,
            verbose=verbose,
        )
        total_saved.extend(saved)

    print(f"\nDone. {len(total_saved)} chart(s) saved to '{output_dir}/'")


if __name__ == '__main__':
    main()