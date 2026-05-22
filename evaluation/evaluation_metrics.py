from typing import List

import numpy as np


def evaluate_summary(
    predicted_summary: np.ndarray,
    user_summary: np.ndarray,
    eval_method: str,
) -> float:
    user_summary = np.atleast_2d(user_summary)
    max_len = max(len(predicted_summary), user_summary.shape[1])

    S = np.zeros(max_len, dtype=int)
    G = np.zeros(max_len, dtype=int)
    S[:len(predicted_summary)] = predicted_summary

    f_scores: List[float] = []
    for user in range(user_summary.shape[0]):
        G[:user_summary.shape[1]] = user_summary[user]
        overlapped = S & G

        sum_s = int(S.sum())
        sum_g = int(G.sum())

        precision = overlapped.sum() / sum_s if sum_s > 0 else 0.0
        recall = overlapped.sum() / sum_g if sum_g > 0 else 0.0

        if precision + recall == 0:
            f_scores.append(0.0)
        else:
            f_scores.append(2.0 * precision * recall * 100.0 / (precision + recall))

    return max(f_scores) if eval_method == 'max' else sum(f_scores) / len(f_scores)