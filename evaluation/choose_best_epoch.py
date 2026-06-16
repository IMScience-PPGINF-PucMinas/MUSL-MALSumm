# -*- coding: utf-8 -*-
import torch
import numpy as np
import csv
import json
import sys

exp_path = sys.argv[1]
dataset  = sys.argv[2]

def train_logs(log_file):
    losses = {}
    losses_names = []

    with open(log_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for (i, row) in enumerate(csv_reader):
            if i == 0:
                for col in range(len(row)):
                    losses[row[col]] = []
                    losses_names.append(row[col])
            else:
                for col in range(len(row)):
                    losses[losses_names[col]].append(float(row[col]))

    loss = losses["loss_epoch"]

    START_EPOCH, tol = 15, 1
    cand_epoch, cand_val = 0, 0
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

    criterion  = torch.tensor(loss)
    argmin_epoch = torch.argmin(criterion).item()
    argmin_diff  = (loss[argmin_epoch] - loss[argmin_epoch - 1]) / loss[argmin_epoch] * 100

    epoch = argmin_epoch if abs(argmin_diff) < abs(cand_val) else cand_epoch
    return epoch + 1

N_SPLITS = 5

per_split_fscores = {}
selected_epochs   = {}

for split in range(N_SPLITS):
    results_file = f"{exp_path}/{dataset}/results/split{split}/f_scores.txt"
    log_file     = f"{exp_path}/{dataset}/logs/split{split}/scalars.csv"

    with open(results_file) as f:
        content = f.read().strip()
        f_scores = json.loads(content) if not "\n" in content else [float(x) for x in content.splitlines()]

    f_scores = [float(v) for v in f_scores]
    per_split_fscores[split] = f_scores

    epoch = train_logs(log_file)
    selected_epochs[split] = epoch

    fscore_at_epoch = np.round(f_scores[epoch], 2)
    print(f"Split {split} → criterion epoch: {epoch}  F-score: {fscore_at_epoch:.2f}%")

criterion_fscores = np.array([
    per_split_fscores[s][selected_epochs[s]] for s in range(N_SPLITS)
])
print(f"\n[Criterion] Average F-score (best epoch per split): {np.mean(criterion_fscores):.2f}%")

best_per_split = np.array([
    max(per_split_fscores[s]) for s in range(N_SPLITS)
])
best_epochs_per_split = [
    int(np.argmax(per_split_fscores[s])) for s in range(N_SPLITS)
]
for s in range(N_SPLITS):
    print(f"  Split {s}: best epoch = {best_epochs_per_split[s]}  F-score = {best_per_split[s]:.2f}%")
print(f"\n[Upper bound] Average F-score (independent best epoch/split): {np.mean(best_per_split):.2f}%")

n_epochs = min(len(per_split_fscores[s]) for s in range(N_SPLITS))

avg_fscore_per_epoch = np.array([
    np.mean([per_split_fscores[s][e] for s in range(N_SPLITS)])
    for e in range(n_epochs)
])

best_global_epoch  = int(np.argmax(avg_fscore_per_epoch))
best_global_fscore = avg_fscore_per_epoch[best_global_epoch]

print(f"\n[Global best epoch] Epoch {best_global_epoch}  "
      f"Avg F-score (all splits): {best_global_fscore:.2f}%")
print("  Per-split F-scores at this epoch:")
for s in range(N_SPLITS):
    fs = per_split_fscores[s][best_global_epoch]
    print(f"    Split {s}: {fs:.2f}%")