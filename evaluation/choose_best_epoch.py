import csv
import json
import sys
from typing import List

import numpy as np
import torch


def _read_loss_from_csv(log_file: str) -> List[float]:
    losses: dict = {}
    names: List[str] = []

    with open(log_file) as f:
        reader = csv.reader(f, delimiter=',')
        for i, row in enumerate(reader):
            if i == 0:
                names = row
                losses = {name: [] for name in names}
            else:
                for col, name in enumerate(names):
                    losses[name].append(float(row[col]))

    return losses['loss_epoch']


def choose_best_epoch(log_file: str) -> int:
    loss = _read_loss_from_csv(log_file)

    START_EPOCH, tol = 15, 1
    cand_epoch, cand_val = 0, 0.0

    for i in range(START_EPOCH, len(loss) - 1):
        diff = (loss[i + 1] - loss[i]) / loss[i + 1] * 100
        if diff <= -tol:
            cand_epoch = i + 1
            cand_val = diff
            break
        if diff >= tol:
            cand_epoch = i
            cand_val = diff
            break

    criterion = torch.tensor(loss)
    argmin_epoch = int(torch.argmin(criterion).item())
    argmin_diff = (loss[argmin_epoch] - loss[argmin_epoch - 1]) / loss[argmin_epoch] * 100

    epoch = argmin_epoch if abs(argmin_diff) < abs(cand_val) else cand_epoch
    return epoch + 1


def main(exp_path: str, dataset: str) -> None:
    all_fscores = np.zeros(5, dtype=float)

    for split in range(5):
        results_file = f'{exp_path}/{dataset}/results/split{split}/f_scores.txt'
        log_file = f'{exp_path}/{dataset}/logs/split{split}/scalars.csv'

        with open(results_file) as f:
            content = f.read().strip()
        f_scores = (
            json.loads(content)
            if not '\n' in content
            else [float(x) for x in content.splitlines()]
        )
        f_scores = [float(x) for x in f_scores]

        selected_epoch = choose_best_epoch(log_file)
        all_fscores[split] = round(f_scores[selected_epoch], 2)
        print(f'Split: {split} -> Criterion Fscore: {all_fscores[split]} @ epoch: {selected_epoch}')

    print(f'Average Fscore: {np.mean(all_fscores):.4f}')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python choose_best_epoch.py <exp_path> <dataset>')
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])