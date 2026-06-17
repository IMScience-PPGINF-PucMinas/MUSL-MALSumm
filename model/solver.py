import os
import json
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm, trange
from .layers.summarizer import xLSTM
from utils.tensorboard_utils import TensorboardWriter
from evaluation.evaluation_metrics import evaluate_summary
from inference.generate_summary import generate_summary


class Solver:
    def __init__(self, config=None, train_loader=None, test_loader=None):
        self.model = None
        self.optimizer = None
        self.writer = None

        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader

        # Accumulated F1 score per epoch — written to f_scores.txt after each evaluate()
        self._f_scores_history = []

        self._set_random_seed()

    def _set_random_seed(self):
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def build(self):
        self._initialize_model()
        self._initialize_optimizer_and_writer()

    def _initialize_model(self):
        self.model = xLSTM(
            input_size=self.config.input_size,
            output_size=self.config.input_size,
            num_segments=self.config.n_segments,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout
        ).to(self.config.device)

        self.model.count_parameters()

        if self.config.init_type is not None:
            self.init_weights(self.model, init_type=self.config.init_type, init_gain=self.config.init_gain)

    def _initialize_optimizer_and_writer(self):
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req
            )
            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        for name, param in net.named_parameters():
            if 'weight' in name and param.dim() >= 2 and "norm" not in name:
                if init_type == "normal":
                    nn.init.normal_(param, mean=0.0, std=init_gain)
                elif init_type == "xavier":
                    nn.init.xavier_uniform_(param, gain=np.sqrt(2.0))
                elif init_type == "kaiming":
                    nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
                elif init_type == "orthogonal":
                    nn.init.orthogonal_(param, gain=np.sqrt(2.0))
                else:
                    raise NotImplementedError(f"Initialization method {init_type} is not implemented.")
            elif 'bias' in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    def train(self):
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            loss_history = self._train_one_epoch(epoch_i)
            self._log_epoch_results(epoch_i, loss_history)
            self.evaluate(epoch_i)

    def _train_one_epoch(self, epoch_i):
        loss_history = []
        dataset = self.train_loader.dataset
        n = len(dataset)

        indices = torch.randperm(n).tolist()

        for batch_start in trange(0, n, self.config.batch_size, desc='Batch', ncols=80, leave=False):
            batch_indices = indices[batch_start: batch_start + self.config.batch_size]
            self.optimizer.zero_grad()
            batch_loss = self._process_batch_indices(dataset, batch_indices)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

        return loss_history

    def _process_batch_indices(self, dataset, indices):
        batch_loss = 0
        for idx in indices:
            frame_features, target = dataset[idx]
            frame_features = frame_features.to(self.config.device)
            target = target.to(self.config.device)

            output, _ = self.model(frame_features.squeeze(0))
            output_adjusted = output.squeeze(0)

            loss = nn.MSELoss()(output_adjusted, target.squeeze(0))
            loss.backward()
            batch_loss += loss.item()

        return batch_loss / len(indices)

    def _log_epoch_results(self, epoch_i, loss_history):
        """Log results for the current epoch."""
        mean_loss = np.mean(loss_history)
        print(f"Epoch {epoch_i} loss: {mean_loss:.4f}")

        if self.config.verbose:
            tqdm.write('Plotting...')

        self.writer.update_loss(mean_loss, epoch_i, 'loss_epoch')
        self._save_checkpoint(epoch_i)

    def _save_checkpoint(self, epoch_i):
        """Save model checkpoint."""
        os.makedirs(self.config.save_dir, exist_ok=True)
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch_i}.pkl')
        tqdm.write(f'Saving parameters at {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def evaluate(self, epoch_i, save_weights=False):
        """Run evaluation for one epoch.

        In addition to saving per-video scores (existing behaviour), computes the
        mean F1-score over all test videos and appends it to f_scores.txt inside
        the checkpoint directory. inference.py reads this file in its fast path to
        select the best epoch without scanning all checkpoints.
        """
        self.model.eval()
        out_scores_dict = {}
        weights_save_path = os.path.join(self.config.score_dir, "weights.h5")

        all_scores    = []
        all_shot_bounds = []
        all_nframes   = []
        all_positions = []
        all_user_summaries = []

        for idx in trange(len(self.test_loader), desc='Evaluate', ncols=80, leave=False):
            frame_features, video_name = self.test_loader[idx]
            scores, attn_weights = self._evaluate_video(frame_features)
            out_scores_dict[video_name] = scores

            if save_weights:
                self._save_attention_weights(weights_save_path, video_name, epoch_i, attn_weights)

            # Collect data needed for F1 computation
            video_data = self._load_video_eval_data(video_name)
            if video_data is not None:
                sb, n_frames, positions, user_summary = video_data
                all_scores.append(scores)
                all_shot_bounds.append(sb)
                all_nframes.append(n_frames)
                all_positions.append(positions)
                all_user_summaries.append(user_summary)

        self._save_scores(out_scores_dict, epoch_i)

        # Compute and persist mean F1 for this epoch
        mean_f1 = self._compute_and_log_f1(
            epoch_i, all_scores, all_shot_bounds, all_nframes,
            all_positions, all_user_summaries,
        )
        return mean_f1

    def _load_video_eval_data(self, video_name):
        """Load shot boundaries, n_frames, positions, and user_summary for a video.

        Returns None if the dataset file or video key is unavailable, so that F1
        computation degrades gracefully rather than crashing the training loop.
        """
        dataset_attr = getattr(self.config, 'dataset_path', None) \
                    or getattr(self.config, 'data_path', None)
        if dataset_attr is None:
            return None

        try:
            with h5py.File(dataset_attr, 'r') as hdf:
                if video_name not in hdf:
                    return None

                sb = np.array(hdf[f"{video_name}/change_points"])

                dataset_name = getattr(self.config, 'dataset', '').lower()
                if dataset_name in ('summe', 'tvsum'):
                    user_summary = np.array(hdf[f"{video_name}/user_summary"])
                    n_frames     = int(np.array(hdf[f"{video_name}/n_frames"]))
                    positions    = np.array(hdf[f"{video_name}/picks"])
                elif dataset_name == 'mrhisum':
                    user_summary = np.array(hdf[f"{video_name}/gt_summary"])
                    n_frames     = int(np.array(hdf[f"{video_name}/features"]).shape[0])
                    positions    = np.arange(n_frames, dtype=int)
                else:
                    return None

            return sb, n_frames, positions, user_summary
        except Exception:
            return None

    def _compute_and_log_f1(self, epoch_i, all_scores, all_shot_bounds,
                             all_nframes, all_positions, all_user_summaries):
        """Compute mean F1 over all evaluated videos, log to TensorBoard, and
        append to f_scores.txt so that inference.py can find the best epoch.

        :param int epoch_i: Current epoch index.
        :return float: Mean F1-score (0–100 scale), or NaN if no videos were evaluated.
        """
        if not all_scores:
            return float('nan')

        eval_method = getattr(self.config, 'eval_method', 'max')

        summaries = generate_summary(all_shot_bounds, all_scores, all_nframes, all_positions)

        f_scores = []
        for summary, user_summary in zip(summaries, all_user_summaries):
            f_scores.append(evaluate_summary(summary, user_summary, eval_method))

        mean_f1 = float(np.nanmean(f_scores))
        tqdm.write(f'Epoch {epoch_i} — mean F1: {mean_f1:.2f}%')

        # Log to TensorBoard
        if self.writer is not None:
            self.writer.update_loss(mean_f1, epoch_i, 'f1_epoch')

        # Persist to f_scores.txt (one float per line, index == epoch number)
        self._f_scores_history.append(mean_f1)
        os.makedirs(self.config.save_dir, exist_ok=True)
        fscores_path = os.path.join(self.config.save_dir, 'f_scores.txt')
        with open(fscores_path, 'w') as fp:
            json.dump(self._f_scores_history, fp)

        # Keep best_model.pkl pointing to the highest-F1 checkpoint so that the
        # inference fast path can load it directly without scanning all epochs.
        best_epoch = int(np.argmax(self._f_scores_history))
        best_src   = os.path.join(self.config.save_dir, f'epoch-{best_epoch}.pkl')
        best_dst   = os.path.join(self.config.save_dir, 'best_model.pkl')
        if os.path.exists(best_src):
            import shutil
            shutil.copy2(best_src, best_dst)

        return mean_f1

    def _evaluate_video(self, frame_features):
        """Evaluate a single video."""
        frame_features = frame_features.view(-1, self.config.input_size).to(self.config.device)
        with torch.no_grad():
            scores, attn_weights = self.model(frame_features)
            scores = scores.squeeze(0).cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(self, weights_save_path, video_name, epoch_i, attn_weights):
        """Save attention weights."""
        with h5py.File(weights_save_path, 'a') as weights:
            weights.create_dataset(f"{video_name}/epoch_{epoch_i}", data=attn_weights)

    def _save_scores(self, out_scores_dict, epoch_i):
        """Save per-video importance scores for the current epoch."""
        os.makedirs(self.config.score_dir, exist_ok=True)

        scores_save_path = os.path.join(
            self.config.score_dir, f"{self.config.video_type}_{epoch_i}.json"
        )
        with open(scores_save_path, 'w') as f:
            if self.config.verbose:
                tqdm.write(f'Saving scores at {scores_save_path}')
            json.dump(out_scores_dict, f)


if __name__ == '__main__':
    pass