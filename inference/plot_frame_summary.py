# -*- coding: utf-8 -*-
"""
plot_frame_summary.py
=====================
Visualises the frame-level binary summary produced by the knapsack selection,
overlaid on the ground-truth annotation votes.

Designed to integrate with inference.py: call ``plot_summary`` directly
after ``generate_summary`` / ``run_inference``, passing the signals that
are already computed there.

Standalone usage (Colab / notebook):
    from plot_frame_summary import plot_summary, predicted_from_indices
    plot_summary(gt_votes, predicted_array, video_name="video_6",
                 save_path="output.jpg")
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from statistics import mean as stat_mean

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FPS: int = 30   # nominal frame rate used for fps-reduction helpers


# ---------------------------------------------------------------------------
# Core plot
# ---------------------------------------------------------------------------

def plot_summary(
    groundtruth: np.ndarray,
    predicted: np.ndarray,
    *,
    video_name: str = "",
    f_score: float | None = None,
    kendall: float | None = None,
    spearman: float | None = None,
    save_path: str = "/content/output.jpg",
) -> None:
    """Plot frame-level ground-truth votes vs predicted binary summary.

    Parameters
    ----------
    groundtruth : 1-D array, length = n_frames
        Accumulated annotator votes per frame (or binary GT).
    predicted : 1-D array, length = n_frames
        Binary array where 1 = frame selected by knapsack.
    video_name : str, optional
        Used as the chart title.
    f_score : float, optional
        F1 score (0-100) — displayed in the title when provided.
    kendall : float, optional
        Kendall τ — displayed in the title when provided.
    spearman : float, optional
        Spearman ρ — displayed in the title when provided.
    save_path : str
        Output image path.
    """
    groundtruth = np.asarray(groundtruth, dtype=np.float32)
    predicted   = np.asarray(predicted,   dtype=np.float32)

    if len(groundtruth) != len(predicted):
        raise ValueError(
            f"Length mismatch: groundtruth={len(groundtruth)}, "
            f"predicted={len(predicted)}"
        )

    n_frames = len(groundtruth)

    # --- Figure sizing: clamp width to a readable range -----------------
    fig_w = float(np.clip(n_frames / 250, 8, 32))
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    # --- Palette --------------------------------------------------------
    BASE_COLOR      = "#7FDBFF"   # light blue  — ground truth
    HIGHLIGHT_COLOR = "#FF4136"   # red         — predicted selection

    ind   = np.arange(n_frames)
    width = 1.0

    bars = ax.bar(ind, groundtruth, width,
                  color=BASE_COLOR, alpha=0.7, edgecolor="none")

    # Colour predicted frames red
    predicted_mask = predicted == 1
    for i in np.where(predicted_mask)[0]:
        bars[i].set_color(HIGHLIGHT_COLOR)
        bars[i].set_alpha(0.9)

    # --- Title ----------------------------------------------------------
    title_parts = [video_name] if video_name else []
    if f_score is not None:
        title_parts.append(f"F1={f_score:.2f}%")
    if kendall is not None:
        title_parts.append(f"τ={kendall:.4f}")
    if spearman is not None:
        title_parts.append(f"ρ={spearman:.4f}")
    if title_parts:
        ax.set_title("  |  ".join(title_parts), fontsize=13, pad=6)

    # --- Axes labels ----------------------------------------------------
    ax.set_xlabel("Frame Index", fontsize=18)
    ax.set_ylabel("Votes",       fontsize=18)
    ax.set_xticks([])
    ax.set_yticks([])

    # --- Legend ---------------------------------------------------------
    legend_elements = [
        Patch(facecolor=BASE_COLOR,      label="Ground Truth", alpha=0.7),
        Patch(facecolor=HIGHLIGHT_COLOR, label="Predicted",    alpha=0.9),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=14)

    # --- Save -----------------------------------------------------------
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ---------------------------------------------------------------------------
# Array construction helpers
# ---------------------------------------------------------------------------

def predicted_from_indices(
    video_len: int,
    selected_indices: list[int],
) -> np.ndarray:
    """Build a binary frame array from a list of selected frame indices.

    Parameters
    ----------
    video_len : int
        Total number of frames in the video.
    selected_indices : list[int]
        Frame indices that were selected (e.g. from knapsack output).

    Returns
    -------
    np.ndarray, dtype float32, shape (video_len,)
    """
    arr = np.zeros(video_len, dtype=np.float32)
    for idx in selected_indices:
        if 0 <= idx < video_len:
            arr[idx] = 1.0
    return arr


def reduce_to_1fps_mean(selected_frames: list[int], fps: int = FPS) -> list[int]:
    """Reduce a list of selected frame indices to one representative per second.

    Groups frames by second (``frame // fps``) and picks the frame closest
    to the group mean.

    Parameters
    ----------
    selected_frames : list[int]
    fps : int

    Returns
    -------
    list[int]
    """
    if not selected_frames:
        return []

    grouped: dict[int, list[int]] = {}
    for f in sorted(selected_frames):
        grouped.setdefault(f // fps, []).append(f)

    return [
        min(group, key=lambda x: abs(x - int(round(stat_mean(group)))))
        for group in (grouped[sec] for sec in sorted(grouped))
    ]


# ---------------------------------------------------------------------------
# Integration helper: call from inference.py after run_inference
# ---------------------------------------------------------------------------

def plot_from_inference(
    video_id: str,
    user_summary: np.ndarray,
    summary: np.ndarray,
    *,
    video_name: str = "",
    f_score: float | None = None,
    kendall: float | None = None,
    spearman: float | None = None,
    output_dir: str = "plots/frame_summary",
) -> str:
    """Convenience wrapper for use inside ``run_inference``.

    Aggregates annotator votes from ``user_summary`` and calls
    ``plot_summary`` with the binary ``summary`` produced by
    ``generate_summary``.

    Parameters
    ----------
    video_id : str
        Used to build the output filename.
    user_summary : ndarray [n_annotators, n_frames] or [n_frames]
        Ground-truth annotation matrix.
    summary : ndarray [n_frames]
        Binary knapsack summary (0/1).
    video_name : str, optional
        Human-readable name for the title.
    f_score, kendall, spearman : float, optional
        Metrics computed by ``run_inference`` — passed through to the title.
    output_dir : str
        Directory where the PNG is saved.

    Returns
    -------
    str: path of the saved file.
    """
    gt_votes = np.atleast_2d(user_summary).sum(axis=0).astype(np.float32)

    label     = video_name or video_id
    safe_name = label.strip().replace(" ", "_")
    save_path = os.path.join(output_dir, f"{video_id}_{safe_name}.png")

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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _load_scores_json(path: str) -> "dict[str, list[float]]":
    """Load a scores JSON file produced by inference.py.

    Accepts two formats:
      - {video_id: [[score], [score], ...]}   ← inference.py default
      - {video_id: [score, score, ...]}        ← flat list variant
    Returns {video_id: [score, ...]} always flat.
    """
    import json
    with open(path) as fp:
        raw = json.load(fp)
    flat: dict[str, list[float]] = {}
    for vid, scores in raw.items():
        if scores and isinstance(scores[0], list):
            flat[vid] = [s[0] for s in scores]
        else:
            flat[vid] = [float(s) for s in scores]
    return flat


def _knapsack(W, wt, val, n):
    """0/1 knapsack — self-contained, no project imports needed."""
    K = [[0.0] * (W + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for w in range(W + 1):
            if wt[i - 1] <= w:
                K[i][w] = max(val[i - 1] + K[i - 1][w - wt[i - 1]], K[i - 1][w])
            else:
                K[i][w] = K[i - 1][w]
    selected, w = [], W
    for i in range(n, 0, -1):
        if K[i][w] != K[i - 1][w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]
    return selected


def _scores_to_summary(scores, h5_path, video_id, gt_key):
    """Run knapsack selection on raw model scores from a results JSON.

    Self-contained: reads h5 data and reproduces generate_summary logic
    without importing anything from the project tree.

    Returns
    -------
    summary      : ndarray [n_frames]  binary knapsack selection (int8)
    user_summary : ndarray [n_annotators, n_frames]
    n_frames     : int
    video_name   : str
    """
    import h5py

    with h5py.File(h5_path, "r") as hf:
        sb           = np.array(hf[f"{video_id}/change_points"])
        n_frames     = int(np.array(hf[f"{video_id}/n_frames"]))
        positions    = np.array(hf[f"{video_id}/picks"])
        user_summary = np.array(hf[f"{video_id}/{gt_key}"])
        video_name   = video_id
        if f"{video_id}/video_name" in hf:
            video_name = str(
                np.array(hf[f"{video_id}/video_name"]).astype(str, copy=False)
            )

    frame_init_scores = np.array(scores, dtype=np.float32)

    # Upsample sub-sampled scores to full frame resolution
    frame_scores = np.zeros(n_frames, dtype=np.float32)
    pos = positions.astype(np.int32)
    if pos[-1] != n_frames:
        pos = np.concatenate([pos, [n_frames]])
    for i in range(len(pos) - 1):
        v = frame_init_scores[i] if i < len(frame_init_scores) else 0.0
        frame_scores[pos[i]:pos[i + 1]] = v

    # Shot-level importance scores
    shot_lengths    = [int(s[1] - s[0] + 1) for s in sb]
    shot_imp_scores = [float(frame_scores[s[0]:s[1] + 1].mean()) for s in sb]

    # Knapsack: budget = 15% of total frames
    final_max_length = int((sb[-1][1] + 1) * 0.15)
    selected = _knapsack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

    # Build binary summary vector
    summary = np.zeros(sb[-1][1] + 1, dtype=np.int8)
    for idx in selected:
        summary[sb[idx][0]:sb[idx][1] + 1] = 1

    return summary, user_summary, n_frames, video_name


def main() -> None:
    """Command-line interface for plot_frame_summary.py.

    The primary input is a results JSON produced by inference.py, whose path
    follows the convention:

        <base>/Summaries/xLSTM/<Dataset>/results/split<N>/<Dataset>_<epoch>.json

    Examples
    --------
    # Plot all videos from a results JSON (auto-runs knapsack to get the summary):
    python plot_frame_summary.py \\
        --h5      /data/eccv16_dataset_summe_google_pool5.h5 \\
        --scores  Summaries/xLSTM/SumMe/results/split0/SumMe_101.json \\
        --dataset summe \\
        --output  plots/

    # Single video only:
    python plot_frame_summary.py \\
        --h5      /data/eccv16_dataset_summe_google_pool5.h5 \\
        --scores  Summaries/xLSTM/SumMe/results/split0/SumMe_101.json \\
        --video   video_10 \\
        --dataset summe \\
        --output  plots/

    # Pass selected frame indices directly (skip knapsack, no scores JSON needed):
    python plot_frame_summary.py \\
        --h5      /data/eccv16_dataset_tvsum_google_pool5.h5 \\
        --video   video_6 \\
        --indices "1484,1485,1486,2000,3100" \\
        --dataset tvsum \\
        --output  plots/

    # Read dataset and epoch from the scores path automatically:
    python plot_frame_summary.py \\
        --h5      /data/eccv16_dataset_summe_google_pool5.h5 \\
        --scores  Summaries/xLSTM/SumMe/results/split0/SumMe_101.json
        # dataset = SumMe (parsed from filename), epoch = 101 (shown in title)
    """
    import argparse
    import json
    import re
    import h5py

    parser = argparse.ArgumentParser(
        description=(
            "Plot frame-level ground-truth vs knapsack-selected summary.\n"
            "Reads model scores from a results JSON produced by inference.py\n"
            "and re-runs generate_summary to obtain the binary selection."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=main.__doc__,
    )

    # --- Required ---
    parser.add_argument(
        "--h5", required=True, metavar="PATH",
        help="Path to the .h5 dataset file (TVSum / SumMe / MrHiSum).",
    )

    # --- Predicted summary source (mutually exclusive) ---
    src = parser.add_mutually_exclusive_group()
    src.add_argument(
        "--scores", metavar="PATH",
        help=(
            "Path to a results JSON produced by inference.py, e.g. "
            "Summaries/xLSTM/SumMe/results/split0/SumMe_101.json. "
            "Dataset name and epoch are parsed from the filename automatically."
        ),
    )
    src.add_argument(
        "--indices", metavar="INTS",
        help="Comma-separated selected frame indices. Requires --video.",
    )

    # --- Video selection ---
    parser.add_argument(
        "--video", metavar="VIDEO_ID", default=None,
        help="Single video key to plot (e.g. 'video_10'). "
             "Omit to plot every video present in the scores JSON.",
    )

    # --- Dataset / ground-truth key ---
    parser.add_argument(
        "--dataset", metavar="NAME", default=None,
        choices=["tvsum", "summe", "mrhisum"],
        help=(
            "Dataset name. Determines the ground-truth h5 key "
            "('user_summary' for tvsum/summe, 'gt_summary' for mrhisum). "
            "Auto-detected from the scores filename when omitted."
        ),
    )

    # --- Optional metrics for the chart title ---
    parser.add_argument("--f_score",  type=float, default=None, metavar="FLOAT",
                        help="F1 score to show in the chart title.")
    parser.add_argument("--kendall",  type=float, default=None, metavar="FLOAT",
                        help="Kendall τ to show in the chart title.")
    parser.add_argument("--spearman", type=float, default=None, metavar="FLOAT",
                        help="Spearman ρ to show in the chart title.")

    # --- Output ---
    parser.add_argument(
        "--output", metavar="DIR", default="plots/frame_summary",
        help="Output directory for PNG files. Created if absent. "
             "Default: plots/frame_summary/",
    )
    parser.add_argument(
        "--fps", type=int, default=FPS, metavar="N",
        help=f"FPS used for the 1-fps reduction stats. Default: {FPS}.",
    )

    args = parser.parse_args()

    # --- Validation ---
    if args.indices and not args.video:
        parser.error("--indices requires --video.")
    if not args.scores and not args.indices:
        parser.error("Provide either --scores or --indices.")

    # --- Auto-detect dataset and epoch from scores filename ---
    epoch: int | None = None
    dataset_name: str = args.dataset or "tvsum"

    if args.scores:
        fname = os.path.basename(args.scores)                 # e.g. SumMe_101.json
        stem  = os.path.splitext(fname)[0]                    # SumMe_101
        parts = stem.split("_")
        if args.dataset is None:
            detected = parts[0].lower()                        # summe / tvsum / mrhisum
            if detected in ("summe", "tvsum", "mrhisum"):
                dataset_name = detected
                print(f"[auto] dataset = {dataset_name}")
        epoch_match = re.search(r"_(\d+)$", stem)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            print(f"[auto] epoch   = {epoch}")

    gt_key = "gt_summary" if dataset_name == "mrhisum" else "user_summary"

    # --- Load scores JSON ---
    if args.scores:
        all_scores = _load_scores_json(args.scores)
        videos = [args.video] if args.video else sorted(all_scores.keys())
    else:
        all_scores = {}
        videos = [args.video]

    print(f"\nH5      : {args.h5}")
    print(f"Videos  : {len(videos)}")
    print(f"Output  : {args.output}\n")

    # --- Process each video ---
    for video_id in videos:
        # --- Branch 1: scores JSON → re-run knapsack via generate_summary ---
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

        # --- Branch 2: explicit frame indices ---
        else:
            import h5py
            with h5py.File(args.h5, "r") as hf:
                if video_id not in hf:
                    print(f"[SKIP] '{video_id}' not in h5 — skipping.")
                    continue
                user_summary = np.array(hf[f"{video_id}/{gt_key}"])
                video_len    = int(np.atleast_2d(user_summary).shape[1])
                video_name   = video_id
                if f"{video_id}/video_name" in hf:
                    video_name = str(
                        np.array(hf[f"{video_id}/video_name"]).astype(str, copy=False)
                    )
            selected  = [int(x.strip()) for x in args.indices.split(",") if x.strip()]
            predicted = predicted_from_indices(video_len, selected)

        # --- Build title: include epoch when parsed from filename ---
        display_name = video_name
        if epoch is not None:
            display_name = f"{video_name}  [epoch {epoch}]"

        # Build output path
        safe_name = video_name.strip().replace(" ", "_")
        save_path = os.path.join(args.output, f"{video_id}_{safe_name}.png")

        # Plot
        plot_summary(
            groundtruth=np.atleast_2d(user_summary).sum(axis=0).astype(np.float32),
            predicted=predicted,
            video_name=display_name,
            f_score=args.f_score,
            kendall=args.kendall,
            spearman=args.spearman,
            save_path=save_path,
        )

        # Stats
        selected_count = int(predicted.sum())
        reduced        = reduce_to_1fps_mean(
            [i for i, v in enumerate(predicted) if v == 1], fps=args.fps
        )
        print(
            f"  {video_id} ({video_name}) | len={video_len} | "
            f"selected={selected_count} ({selected_count/video_len*100:.1f}%) | "
            f"budget_15%={int(video_len*0.15)} | "
            f"1fps={len(reduced)}"
        )


if __name__ == "__main__":
    main()
