import argparse
import json
import os
import re
from statistics import mean as stat_mean
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

FPS: int = 30


def plot_summary(
    groundtruth: np.ndarray,
    predicted: np.ndarray,
    video_name: str = '',
    f_score: Optional[float] = None,
    kendall: Optional[float] = None,
    spearman: Optional[float] = None,
    save_path: str = '/content/output.jpg',
) -> None:
    groundtruth = np.asarray(groundtruth, dtype=np.float32)
    predicted = np.asarray(predicted, dtype=np.float32)

    if len(groundtruth) != len(predicted):
        raise ValueError(
            f'Length mismatch: groundtruth={len(groundtruth)}, predicted={len(predicted)}'
        )

    n_frames = len(groundtruth)
    fig_w = float(np.clip(n_frames / 250, 8, 32))
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    BASE_COLOR = '#7FDBFF'
    HIGHLIGHT_COLOR = '#FF4136'

    ind = np.arange(n_frames)
    bars = ax.bar(ind, groundtruth, 1.0, color=BASE_COLOR, alpha=0.7, edgecolor='none')

    for i in np.where(predicted == 1)[0]:
        bars[i].set_color(HIGHLIGHT_COLOR)
        bars[i].set_alpha(0.9)

    title_parts = [video_name] if video_name else []
    if f_score is not None:
        title_parts.append(f'F1={f_score:.2f}%')
    if kendall is not None:
        title_parts.append(f'τ={kendall:.4f}')
    if spearman is not None:
        title_parts.append(f'ρ={spearman:.4f}')
    if title_parts:
        ax.set_title('  |  '.join(title_parts), fontsize=13, pad=6)

    ax.set_xlabel('Frame Index', fontsize=18)
    ax.set_ylabel('Votes', fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.legend(
        handles=[
            Patch(facecolor=BASE_COLOR, label='Ground Truth', alpha=0.7),
            Patch(facecolor=HIGHLIGHT_COLOR, label='Predicted', alpha=0.9),
        ],
        loc='upper right',
        fontsize=14,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f'Saved → {save_path}')


def predicted_from_indices(video_len: int, selected_indices: List[int]) -> np.ndarray:
    arr = np.zeros(video_len, dtype=np.float32)
    for idx in selected_indices:
        if 0 <= idx < video_len:
            arr[idx] = 1.0
    return arr


def reduce_to_1fps_mean(selected_frames: List[int], fps: int = FPS) -> List[int]:
    if not selected_frames:
        return []

    grouped: Dict[int, List[int]] = {}
    for f in sorted(selected_frames):
        grouped.setdefault(f // fps, []).append(f)

    return [
        min(group, key=lambda x: abs(x - int(round(stat_mean(group)))))
        for group in (grouped[sec] for sec in sorted(grouped))
    ]


def plot_from_inference(
    video_id: str,
    user_summary: np.ndarray,
    summary: np.ndarray,
    video_name: str = '',
    f_score: Optional[float] = None,
    kendall: Optional[float] = None,
    spearman: Optional[float] = None,
    output_dir: str = 'plots/frame_summary',
) -> str:
    gt_votes = np.atleast_2d(user_summary).sum(axis=0).astype(np.float32)
    label = video_name or video_id
    safe_name = label.strip().replace(' ', '_')
    save_path = os.path.join(output_dir, f'{video_id}_{safe_name}.png')

    plot_summary(
        gt_votes,
        summary,
        video_name=label,
        f_score=f_score,
        kendall=kendall,
        spearman=spearman,
        save_path=save_path,
    )
    return save_path


def _load_scores_json(path: str) -> Dict[str, List[float]]:
    with open(path) as fp:
        raw = json.load(fp)
    flat: Dict[str, List[float]] = {}
    for vid, scores in raw.items():
        if scores and isinstance(scores[0], list):
            flat[vid] = [s[0] for s in scores]
        else:
            flat[vid] = [float(s) for s in scores]
    return flat


def _knapsack(W: int, wt: List[int], val: List[float], n: int) -> List[int]:
    K = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    selected: List[int] = []
    w = W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]
    return selected


def _scores_to_summary(
    scores: List[float],
    h5_path: str,
    video_id: str,
    gt_key: str,
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    with h5py.File(h5_path, 'r') as hf:
        sb = np.array(hf[f'{video_id}/change_points'])
        n_frames = int(np.array(hf[f'{video_id}/n_frames']))
        positions = np.array(hf[f'{video_id}/picks'])
        user_summary = np.array(hf[f'{video_id}/{gt_key}'])
        video_name = video_id
        if f'{video_id}/video_name' in hf:
            video_name = str(np.array(hf[f'{video_id}/video_name']).astype(str, copy=False))

    frame_init_scores = np.array(scores, dtype=np.float32)
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        frame_scores[pos[i]:pos[i + 1]] = frame_init_scores[i] if i < len(frame_init_scores) else 0.0

    shot_lengths = [int(s[1] - s[0] + 1) for s in sb]
    shot_imp_scores = [float(frame_scores[s[0]:s[1] + 1].mean()) for s in sb]
    final_max_length = int((sb[-1][1] + 1) * 0.15)
    selected = _knapsack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    summary = np.zeros(sb[-1][1] + 1, dtype=np.int8)
    for idx in selected:
        summary[sb[idx][0]:sb[idx][1] + 1] = 1

    return summary, user_summary, n_frames, video_name


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Plot frame-level ground-truth vs knapsack-selected summary.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--h5', required=True, metavar='PATH')

    src = parser.add_mutually_exclusive_group()
    src.add_argument('--scores', metavar='PATH')
    src.add_argument('--indices', metavar='INTS')

    parser.add_argument('--video', metavar='VIDEO_ID', default=None)
    parser.add_argument('--dataset', metavar='NAME', default=None,
                        choices=['tvsum', 'summe', 'mrhisum'])
    parser.add_argument('--f_score', type=float, default=None)
    parser.add_argument('--kendall', type=float, default=None)
    parser.add_argument('--spearman', type=float, default=None)
    parser.add_argument('--output', metavar='DIR', default='plots/frame_summary')
    parser.add_argument('--fps', type=int, default=FPS)

    args = parser.parse_args()

    if args.indices and not args.video:
        parser.error('--indices requires --video.')
    if not args.scores and not args.indices:
        parser.error('Provide either --scores or --indices.')

    epoch: Optional[int] = None
    dataset_name: str = args.dataset or 'tvsum'

    if args.scores:
        fname = os.path.basename(args.scores)
        stem = os.path.splitext(fname)[0]
        parts = stem.split('_')
        if args.dataset is None:
            detected = parts[0].lower()
            if detected in ('summe', 'tvsum', 'mrhisum'):
                dataset_name = detected
                print(f'[auto] dataset = {dataset_name}')
        epoch_match = re.search(r'_(\d+)$', stem)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            print(f'[auto] epoch   = {epoch}')

    gt_key = 'gt_summary' if dataset_name == 'mrhisum' else 'user_summary'

    if args.scores:
        all_scores = _load_scores_json(args.scores)
        videos = [args.video] if args.video else sorted(all_scores.keys())
    else:
        all_scores = {}
        videos = [args.video]

    print(f'\nH5      : {args.h5}')
    print(f'Videos  : {len(videos)}')
    print(f'Output  : {args.output}\n')

    for video_id in videos:
        if args.scores:
            if video_id not in all_scores:
                print(f"[SKIP] '{video_id}' not in scores JSON — skipping.")
                continue
            try:
                summary, user_summary, n_frames, video_name = _scores_to_summary(
                    all_scores[video_id], args.h5, video_id, gt_key
                )
            except Exception as exc:
                print(f"[SKIP] '{video_id}' — {exc}")
                continue
            video_len = n_frames
            predicted = summary.astype(np.float32)
        else:
            with h5py.File(args.h5, 'r') as hf:
                if video_id not in hf:
                    print(f"[SKIP] '{video_id}' not in h5 — skipping.")
                    continue
                user_summary = np.array(hf[f'{video_id}/{gt_key}'])
                video_len = int(np.atleast_2d(user_summary).shape[1])
                video_name = video_id
                if f'{video_id}/video_name' in hf:
                    video_name = str(
                        np.array(hf[f'{video_id}/video_name']).astype(str, copy=False)
                    )
            selected = [int(x.strip()) for x in args.indices.split(',') if x.strip()]
            predicted = predicted_from_indices(video_len, selected)

        display_name = video_name if epoch is None else f'{video_name}  [epoch {epoch}]'
        safe_name = video_name.strip().replace(' ', '_')
        save_path = os.path.join(args.output, f'{video_id}_{safe_name}.png')

        plot_summary(
            groundtruth=np.atleast_2d(user_summary).sum(axis=0).astype(np.float32),
            predicted=predicted,
            video_name=display_name,
            f_score=args.f_score,
            kendall=args.kendall,
            spearman=args.spearman,
            save_path=save_path,
        )

        selected_count = int(predicted.sum())
        reduced = reduce_to_1fps_mean(
            [i for i, v in enumerate(predicted) if v == 1], fps=args.fps
        )
        print(
            f'  {video_id} ({video_name}) | len={video_len} | '
            f'selected={selected_count} ({selected_count / video_len * 100:.1f}%) | '
            f'budget_15%={int(video_len * 0.15)} | '
            f'1fps={len(reduced)}'
        )


if __name__ == '__main__':
    main()