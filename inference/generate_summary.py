from typing import List

import numpy as np

from .knapsack_implementation import knapSack


def generate_summary(
    all_shot_bound: List[np.ndarray],
    all_scores: List[np.ndarray],
    all_nframes: List[int],
    all_positions: List[np.ndarray],
) -> List[np.ndarray]:
    all_summaries: List[np.ndarray] = []

    for video_index in range(len(all_scores)):
        shot_bound = all_shot_bound[video_index]
        frame_init_scores = all_scores[video_index]
        n_frames = all_nframes[video_index]
        positions = all_positions[video_index]

        frame_scores = np.zeros(n_frames, dtype=np.float32)
        pos = positions.astype(np.int32)
        if pos[-1] != n_frames:
            pos = np.concatenate([pos, [n_frames]])

        for i in range(len(pos) - 1):
            value = frame_init_scores[i] if i < len(frame_init_scores) else 0.0
            frame_scores[pos[i]:pos[i + 1]] = value

        shot_lengths: List[int] = []
        shot_imp_scores: List[float] = []
        for shot in shot_bound:
            shot_lengths.append(int(shot[1] - shot[0] + 1))
            shot_imp_scores.append(float(frame_scores[shot[0]:shot[1] + 1].mean()))

        final_shot = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)

        selected = knapSack(final_max_length, shot_lengths, shot_imp_scores, len(shot_lengths))

        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot_idx in selected:
            summary[shot_bound[shot_idx][0]:shot_bound[shot_idx][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries