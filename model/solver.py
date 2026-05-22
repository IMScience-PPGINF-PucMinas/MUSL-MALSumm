import os
import json
import random
import copy
from typing import List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from .layers.summarizer import xLSTM
from utils.tensorboard_utils import TensorboardWriter


class Solver:
    def __init__(self, config=None, train_loader=None, test_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model: Optional[xLSTM] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.writer: Optional[TensorboardWriter] = None
        self._set_random_seed()

    def _set_random_seed(self) -> None:
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self) -> None:
        self._initialize_model()
        self._initialize_optimizer_and_writer()

    def _build_model(self) -> xLSTM:
        model = xLSTM(
            input_size=self.config.input_size,
            output_size=self.config.input_size,
            num_segments=self.config.n_segments,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.config.device)
        model.count_parameters()
        if self.config.init_type is not None:
            self._init_weights(model, self.config.init_type, self.config.init_gain)
        return model

    def _initialize_model(self) -> None:
        self.model = self._build_model()

    def _initialize_optimizer_and_writer(self) -> None:
        if self.config.mode == "train":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def _init_weights(net: nn.Module, init_type: str = "xavier", init_gain: float = 1.4142) -> None:
        for name, param in net.named_parameters():
            if "weight" in name and param.dim() >= 2 and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"Initialization method '{init_type}' is not implemented.")
            elif "bias" in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    def train(self) -> None:
        all_samples = list(self.train_loader)
        kf = KFold(n_splits=5, shuffle=True, random_state=self.config.seed)
        indices = np.arange(len(all_samples))

        fold_results: List[dict] = []

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(indices)):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1} / 5")
            print(f"{'='*60}")

            fold_train = [all_samples[i] for i in train_idx]
            fold_val = [all_samples[i] for i in val_idx]

            fold_model = self._build_model()
            fold_optimizer = optim.Adam(
                fold_model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )

            best_val_loss = float("inf")
            best_state = None

            for epoch_i in trange(self.config.n_epochs, desc=f"Fold {fold_idx+1} Epochs", ncols=80):
                train_loss = self._train_one_epoch(fold_model, fold_optimizer, fold_train, epoch_i, fold_idx)
                val_loss = self._validate(fold_model, fold_val)

                tqdm.write(
                    f"[Fold {fold_idx+1}] Epoch {epoch_i:03d} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
                )

                if self.writer is not None:
                    self.writer.update_loss(train_loss, epoch_i + fold_idx * self.config.n_epochs, f"fold_{fold_idx+1}/train_loss")
                    self.writer.update_loss(val_loss, epoch_i + fold_idx * self.config.n_epochs, f"fold_{fold_idx+1}/val_loss")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(fold_model.state_dict())
                    self._save_checkpoint(best_state, fold_idx, epoch_i)

            fold_results.append({"fold": fold_idx + 1, "best_val_loss": best_val_loss})
            print(f"  Best validation loss for fold {fold_idx+1}: {best_val_loss:.4f}")

        self._log_cross_validation_summary(fold_results)

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.evaluate(-1)

    def _train_one_epoch(
        self,
        model: xLSTM,
        optimizer: optim.Optimizer,
        samples: list,
        epoch_i: int,
        fold_idx: int,
    ) -> float:
        model.train()
        loss_history: List[float] = []
        num_batches = len(samples) // self.config.batch_size

        for batch_start in range(0, num_batches * self.config.batch_size, self.config.batch_size):
            optimizer.zero_grad()
            batch = samples[batch_start: batch_start + self.config.batch_size]
            batch_loss = self._process_batch(model, batch)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip)
            optimizer.step()

        return float(np.mean(loss_history)) if loss_history else 0.0

    def _process_batch(self, model: xLSTM, batch: list) -> float:
        total_loss = 0.0
        for frame_features, target in batch:
            frame_features = frame_features.to(self.config.device)
            target = target.to(self.config.device)

            output, _ = model(frame_features.squeeze(0))
            output_adjusted = output.mean(dim=0) if output.dim() > 1 else output
            loss = nn.MSELoss()(output_adjusted, target.squeeze(0))
            loss.backward()
            total_loss += loss.item()

        return total_loss / max(len(batch), 1)

    def _validate(self, model: xLSTM, samples: list) -> float:
        model.eval()
        val_losses: List[float] = []

        with torch.no_grad():
            for frame_features, target in samples:
                frame_features = frame_features.to(self.config.device)
                target = target.to(self.config.device)

                output, _ = model(frame_features.squeeze(0))
                output_adjusted = output.mean(dim=0) if output.dim() > 1 else output
                loss = nn.MSELoss()(output_adjusted, target.squeeze(0))
                val_losses.append(loss.item())

        return float(np.mean(val_losses)) if val_losses else 0.0

    def _save_checkpoint(self, state_dict: dict, fold_idx: int, epoch_i: int) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        ckpt_path = os.path.join(self.config.save_dir, f"fold-{fold_idx+1}_epoch-{epoch_i}.pkl")
        tqdm.write(f"Saving checkpoint at {ckpt_path}")
        torch.save(state_dict, ckpt_path)

    def _log_cross_validation_summary(self, fold_results: List[dict]) -> None:
        val_losses = [r["best_val_loss"] for r in fold_results]
        print(f"\n{'='*60}")
        print("  Cross-Validation Summary")
        print(f"{'='*60}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Best Val Loss = {r['best_val_loss']:.4f}")
        print(f"  Mean Val Loss: {np.mean(val_losses):.4f} ± {np.std(val_losses):.4f}")
        print(f"{'='*60}\n")

        summary_path = os.path.join(self.config.score_dir, "cv_summary.json")
        os.makedirs(self.config.score_dir, exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(
                {
                    "folds": fold_results,
                    "mean_val_loss": float(np.mean(val_losses)),
                    "std_val_loss": float(np.std(val_losses)),
                },
                f,
                indent=2,
            )

    def evaluate(self, epoch_i: int, save_weights: bool = False) -> None:
        self.model.eval()
        out_scores_dict: dict = {}
        weights_save_path = os.path.join(self.config.score_dir, "weights.h5")

        for frame_features, video_name in tqdm(self.test_loader, desc="Evaluate", ncols=80, leave=False):
            scores, attn_weights = self._evaluate_video(frame_features)
            out_scores_dict[video_name] = scores

            if save_weights:
                self._save_attention_weights(weights_save_path, video_name, epoch_i, attn_weights)

        self._save_scores(out_scores_dict, epoch_i)

    def _evaluate_video(self, frame_features: torch.Tensor) -> Tuple[list, np.ndarray]:
        frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
        with torch.no_grad():
            scores, attn_weights = self.model(frame_features)
            scores = (scores.squeeze(0) if scores.dim() > 1 else scores).cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(
        self, path: str, video_name: str, epoch_i: int, attn_weights: np.ndarray
    ) -> None:
        with h5py.File(path, "a") as f:
            f.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)

    def _save_scores(self, out_scores_dict: dict, epoch_i: int) -> None:
        os.makedirs(self.config.score_dir, exist_ok=True)
        scores_save_path = os.path.join(
            self.config.score_dir, f"{self.config.video_type}_{epoch_i}.json"
        )
        with open(scores_save_path, "w") as f:
            if self.config.verbose:
                tqdm.write(f"Saving scores at {scores_save_path}")
            json.dump(out_scores_dict, f)
        os.chmod(scores_save_path, 0o777)