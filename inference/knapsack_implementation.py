# -*- coding: utf-8 -*-
# author: Bhavya Jain
# link: https://github.com/wulfebw/algorithms/blob/master/scripts/dynamic_programming/knapsack.py

import numpy as np


def knapSack(W, wt, val, n):
    wt  = np.array(wt,  dtype=np.int32)
    val = np.array(val, dtype=np.float32)

    K = np.zeros((n + 1, W + 1), dtype=np.float32)

    capacities = np.arange(W + 1, dtype=np.int32)

    for i in range(1, n + 1):
        w_i = wt[i - 1]
        v_i = val[i - 1]

        K[i] = K[i - 1]

        mask      = capacities >= w_i
        prev_caps = np.maximum(capacities - w_i, 0)          # clamp to avoid negative index
        candidate = v_i + K[i - 1, prev_caps]
        K[i]      = np.where(mask, np.maximum(K[i], candidate), K[i])

    selected = []
    w = W
    for i in range(n, 0, -1):
        if K[i, w] != K[i - 1, w]:
            selected.insert(0, i - 1)
            w -= wt[i - 1]

    return selected


if __name__ == "__main__":
    val      = [4, 4, 2, 2, 2, 4]
    wt       = [2, 2, 1, 1, 1, 2]
    W        = 7
    n        = len(val)
    selected = knapSack(W, wt, val, n)
    print("Selected shot indices:", selected)   # expected: [0, 1, 5] or equivalent optimal set