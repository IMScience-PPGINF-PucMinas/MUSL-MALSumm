import copy
import json
import os
import random
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from tqdm import tqdm, trange

from .layers.summarizer import xLSTM
from data.data_loader import VideoRecord
from evaluation.evaluation_metrics import evaluate_summary
from inference.generate_summary import generate_summary
from utils.tensorboard_utils import TensorboardWriter


class Solver:
    def __init__(self, config=None, train_loader=None, test_loader=None):
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model: Optional[xLSTM] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.writer: Optional[TensorboardWriter] = None
        self._eval_method = 'avg' if config.video_type.lower() == 'tvsum' else 'max'
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
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def _init_weights(net: nn.Module, init_type: str = 'xavier', init_gain: float = 1.4142) -> None:
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim() >= 2 and 'norm' not in name:
                if init_type == 'normal':
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == 'kaiming':
                    nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"Initialization method '{init_type}' is not implemented.")
            elif 'bias' in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    def train(self) -> None:
        all_samples: List[VideoRecord] = list(self.train_loader)
        n_samples = len(all_samples)

        effective_batch = min(self.config.batch_size, max(1, n_samples * 4 // 5))
        if effective_batch != self.config.batch_size:
            tqdm.write(
                f'[Warning] batch_size={self.config.batch_size} exceeds fold train size. '
                f'Using batch_size={effective_batch} instead.'
            )

        kf = KFold(n_splits=5, shuffle=True, random_state=self.config.seed)

        fold_results: List[dict] = []
        global_best_fscore = -1.0
        global_best_state: Optional[dict] = None

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(np.arange(n_samples))):
            print(f"\n{'='*60}")
            print(f"  Fold {fold_idx + 1} / 5  "
                  f"(train={len(train_idx)}, val={len(val_idx)})")
            print(f"{'='*60}")

            fold_train = [all_samples[i] for i in train_idx]
            fold_val = [all_samples[i] for i in val_idx]

            fold_model = self._build_model()
            fold_optimizer = optim.Adam(
                fold_model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )

            best_fscore = -1.0
            best_state: Optional[dict] = None

            for epoch_i in trange(self.config.n_epochs, desc=f'Fold {fold_idx+1} Epochs', ncols=80):
                train_loss = self._train_one_epoch(fold_model, fold_optimizer, fold_train, effective_batch)
                val_fscore = self._validate_fscore(fold_model, fold_val)

                tqdm.write(
                    f'[Fold {fold_idx+1}] Epoch {epoch_i:03d} | '
                    f'Train Loss: {train_loss:.4f} | Val F-score: {val_fscore:.2f}%'
                )

                step = epoch_i + fold_idx * self.config.n_epochs
                if self.writer is not None:
                    self.writer.update_loss(train_loss, step, f'fold_{fold_idx+1}/train_loss')
                    self.writer.update_loss(val_fscore, step, f'fold_{fold_idx+1}/val_fscore')

                if val_fscore > best_fscore:
                    best_fscore = val_fscore
                    best_state = copy.deepcopy(fold_model.state_dict())
                    self._save_checkpoint(best_state, epoch_i, val_fscore, fold_idx)

            fold_results.append({'fold': fold_idx + 1, 'best_val_fscore': best_fscore})
            print(f'  Best F-score for fold {fold_idx+1}: {best_fscore:.2f}%')

            if best_fscore > global_best_fscore and best_state is not None:
                global_best_fscore = best_fscore
                global_best_state = best_state

        self._log_cross_validation_summary(fold_results)

        if global_best_state is not None:
            self.model.load_state_dict(global_best_state)
            best_model_path = os.path.join(self.config.save_dir, 'best_model.pkl')
            os.makedirs(self.config.save_dir, exist_ok=True)
            torch.save(global_best_state, best_model_path)
            tqdm.write(
                f'Best model saved at {best_model_path} '
                f'(F-score: {global_best_fscore:.2f}%)'
            )

        self.evaluate(-1)

    def _train_one_epoch(
        self,
        model: xLSTM,
        optimizer: optim.Optimizer,
        samples: List[VideoRecord],
        batch_size: int,
    ) -> float:
        model.train()

        indices = list(range(len(samples)))
        random.shuffle(indices)
        shuffled = [samples[i] for i in indices]

        loss_history: List[float] = []

        for batch_start in range(0, len(shuffled), batch_size):
            batch = shuffled[batch_start: batch_start + batch_size]
            if not batch:
                continue
            optimizer.zero_grad()
            batch_loss = self._process_batch(model, batch)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.clip)
            optimizer.step()

        return float(np.mean(loss_history)) if loss_history else 0.0

    def _process_batch(self, model: xLSTM, batch: List[VideoRecord]) -> float:
        total_loss = 0.0
        for record in batch:
            features = record.features.to(self.config.device)   # (T, C)
            target = record.gtscore.to(self.config.device)      # (T,)

            output, _ = model(features)

            min_len = min(output.shape[0], target.shape[0])
            loss = nn.MSELoss()(output[:min_len], target[:min_len])
            loss.backward()
            total_loss += loss.item()

        return total_loss / len(batch)

    def _validate_fscore(self, model: xLSTM, samples: List[VideoRecord]) -> float:
        model.eval()
        f_scores: List[float] = []

        with torch.no_grad():
            for record in samples:
                features = record.features.to(self.config.device)   # (T, C)

                scores, _ = model(features)                         # (T,)
                scores_list = scores.cpu().numpy().tolist()

                summary = generate_summary(
                    [record.shot_bound],
                    [scores_list],
                    [record.n_frames],
                    [record.positions],
                )[0]

                f_score = evaluate_summary(summary, record.user_summary, self._eval_method)
                f_scores.append(f_score)

        return float(np.mean(f_scores)) if f_scores else 0.0

    def _save_checkpoint(
        self,
        state_dict: dict,
        epoch_i: int,
        fscore: float,
        fold_idx: int,
    ) -> None:
        os.makedirs(self.config.save_dir, exist_ok=True)
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch_i}.pkl')
        torch.save(state_dict, ckpt_path)
        tqdm.write(
            f'Checkpoint saved: {ckpt_path}  '
            f'(fold {fold_idx+1}, F-score: {fscore:.2f}%)'
        )

    def _log_cross_validation_summary(self, fold_results: List[dict]) -> None:
        fscores = [r['best_val_fscore'] for r in fold_results]
        print(f"\n{'='*60}")
        print('  Cross-Validation Summary')
        print(f"{'='*60}")
        for r in fold_results:
            print(f"  Fold {r['fold']}: Best Val F-score = {r['best_val_fscore']:.2f}%")
        print(f'  Mean F-score: {np.mean(fscores):.2f}% ± {np.std(fscores):.2f}%')
        print(f"{'='*60}\n")

        os.makedirs(self.config.score_dir, exist_ok=True)
        with open(os.path.join(self.config.score_dir, 'cv_summary.json'), 'w') as f:
            json.dump(
                {
                    'folds': fold_results,
                    'mean_val_fscore': float(np.mean(fscores)),
                    'std_val_fscore': float(np.std(fscores)),
                },
                f,
                indent=2,
            )

    def evaluate(self, epoch_i: int, save_weights: bool = False) -> None:
        self.model.eval()
        out_scores_dict: dict = {}
        weights_save_path = os.path.join(self.config.score_dir, 'weights.h5')

        for record in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            scores, attn_weights = self._evaluate_video(record.features)
            out_scores_dict[record.video_name] = scores

            if save_weights:
                self._save_attention_weights(
                    weights_save_path, record.video_name, epoch_i, attn_weights
                )

        self._save_scores(out_scores_dict, epoch_i)

    def _evaluate_video(self, frame_features: torch.Tensor) -> Tuple[list, np.ndarray]:
        # Accept (T, C) directly — model handles the batch dim internally
        frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
        with torch.no_grad():
            scores, attn_weights = self.model(frame_features)
            scores = scores.cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(
        self, path: str, video_name: str, epoch_i: int, attn_weights: np.ndarray
    ) -> None:
        with h5py.File(path, 'a') as f:
            f.create_dataset(f'{video_name}/epoch_{epoch_i}', data=attn_weights)

    def _save_scores(self, out_scores_dict: dict, epoch_i: int) -> None:
        os.makedirs(self.config.score_dir, exist_ok=True)
        scores_save_path = os.path.join(
            self.config.score_dir, f'{self.config.video_type}_{epoch_i}.json'
        )
        with open(scores_save_path, 'w') as f:
            if self.config.verbose:
                tqdm.write(f'Saving scores at {scores_save_path}')
            json.dump(out_scores_dict, f)
        os.chmod(scores_save_path, 0o777)