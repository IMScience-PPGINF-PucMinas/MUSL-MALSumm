# -*- coding: utf-8 -*-
import numpy as np
from .knapsack_implementation import knapSack

np.set_printoptions(edgeitems=20)


def expand_scores_to_frames(frame_init_scores, positions, n_frames):
    """Expand sub-sampled importance scores to the full frame sequence via nearest-neighbor
    propagation: each sub-sampled score is broadcast to the interval [positions[i], positions[i+1]).

    Extracted as a standalone utility so that inference.py can call it directly instead of
    replicating the same logic for the Kendall/Spearman correlation computation.

    :param array-like frame_init_scores: Importance scores for the sub-sampled frames (length M).
    :param np.ndarray positions: Indices of the sub-sampled frames in the original sequence.
    :param int n_frames: Total number of frames in the original video.
    :return np.ndarray: Frame-level importance scores of shape (n_frames,), dtype float32.
    """
    frame_init_scores = np.asarray(frame_init_scores, dtype=np.float32)
    positions = np.asarray(positions)
    if positions.dtype != np.int32:
        positions = positions.astype(np.int32)
    if positions[-1] != n_frames:
        positions = np.concatenate([positions, [n_frames]])

    frame_scores = np.zeros(n_frames, dtype=np.float32)
    for i in range(len(positions) - 1):
        pos_left, pos_right = positions[i], positions[i + 1]
        if i < len(frame_init_scores):
            frame_scores[pos_left:pos_right] = frame_init_scores[i]
        # else: remains 0 (out-of-bounds guard, preserves original behaviour)

    return frame_scores


def generate_summary(all_shot_bound, all_scores, all_nframes, all_positions):
    """Generate the automatic machine summary, based on the video shots; the frame importance
    scores; the number of frames in the original video and the position of the sub-sampled frames
    of the original video.

    :param list[np.ndarray] all_shot_bound: The video shots for all the -original- testing videos.
    :param list[np.ndarray] all_scores: The calculated frame importance scores for all the
        sub-sampled testing videos.
    :param list[np.ndarray] all_nframes: The number of frames for all the -original- testing
        videos.
    :param list[np.ndarray] all_positions: The position of the sub-sampled frames for all the
        -original- testing videos.
    :return: A list containing the indices of the selected frames for all the -original- testing
        videos.
    """
    all_summaries = []
    for video_index in range(len(all_scores)):
        shot_bound        = all_shot_bound[video_index]   # [number_of_shots, 2]
        frame_init_scores = all_scores[video_index]
        n_frames          = all_nframes[video_index]
        positions         = all_positions[video_index]

        # Expand sub-sampled scores to the full frame sequence
        frame_scores = expand_scores_to_frames(frame_init_scores, positions, n_frames)

        # Shot-level importance: mean score over all frames in each shot
        shot_lengths    = [int(shot[1] - shot[0] + 1) for shot in shot_bound]
        shot_imp_scores = [float(frame_scores[shot[0]:shot[1] + 1].mean()) for shot in shot_bound]

        # Select the best shots via 0/1 knapsack (budget = 15 % of total frames)
        final_shot       = shot_bound[-1]
        final_max_length = int((final_shot[1] + 1) * 0.15)
        selected         = knapSack(final_max_length, shot_lengths, shot_imp_scores,
                                    len(shot_lengths))

        # Build binary summary vector
        summary = np.zeros(final_shot[1] + 1, dtype=np.int8)
        for shot in selected:
            summary[shot_bound[shot][0]:shot_bound[shot][1] + 1] = 1

        all_summaries.append(summary)

    return all_summaries