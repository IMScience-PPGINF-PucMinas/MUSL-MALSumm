import os
import json
import random
import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm, trange
from .layers.summarizer import xLSTM
from utils.tensorboard_utils import TensorboardWriter


def ranking_loss(scores, targets, margin=1.0):
    """Pairwise ranking loss: penalizes when a frame with higher target score
    receives a lower predicted score than a frame with lower target score.

    Args:
        scores:  [T] predicted importance scores
        targets: [T] ground-truth importance scores (normalized 0-1)
        margin:  minimum desired score gap between correctly ordered pairs
    Returns:
        Scalar loss value.
    """
    # [T, T] — diff_target[i,j] > 0 means frame i is more important than j
    diff_target = targets.unsqueeze(1) - targets.unsqueeze(0)
    diff_score  = scores.unsqueeze(1)  - scores.unsqueeze(0)
    mask = (diff_target > 0).float()
    loss = mask * F.relu(margin - diff_score)
    return loss.mean()


class Solver:
    def __init__(self, config=None, train_loader=None, test_loader=None):
        self.model     = None
        self.optimizer = None
        self.scheduler = None
        self.writer    = None

        self.config       = config
        self.train_loader = train_loader
        self.test_loader  = test_loader

        self._set_random_seed()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_random_seed(self):
        """Set random seed for reproducibility."""
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

    def build(self):
        self._initialize_model()
        self._initialize_optimizer_and_writer()

    def _initialize_model(self):
        """Initialize the xLSTM model."""
        self.model = xLSTM(
            input_size=self.config.input_size,
            output_size=self.config.input_size,
            num_segments=self.config.n_segments,
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        ).to(self.config.device)

        if self.config.init_type is not None:
            self.init_weights(
                self.model,
                init_type=self.config.init_type,
                init_gain=self.config.init_gain,
            )

    def _initialize_optimizer_and_writer(self):
        """Initialize the optimizer, LR scheduler and Tensorboard writer."""
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )

            # FIX 4 — LR scheduler: reduz lr quando a loss estagna,
            # evitando oscilações nas épocas finais com lr fixo.
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
            )

            self.writer = TensorboardWriter(str(self.config.log_dir))

    @staticmethod
    def init_weights(net, init_type="xavier", init_gain=1.4142):
        """Initialize model weights."""
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
                    raise NotImplementedError(
                        f"Initialization method {init_type} is not implemented."
                    )
            elif 'bias' in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """Train the model."""
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            loss_history = self._train_one_epoch(epoch_i)
            mean_loss = self._log_epoch_results(epoch_i, loss_history)

            # FIX 4 — atualiza o scheduler com a loss média da época
            if self.scheduler is not None:
                self.scheduler.step(mean_loss)

            self.evaluate(epoch_i)

    def _train_one_epoch(self, epoch_i):
        """Train the model for one epoch."""
        loss_history = []
        num_batches  = len(self.train_loader) // self.config.batch_size
        iterator     = iter(self.train_loader)

        for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
            self.optimizer.zero_grad()
            batch_loss = self._process_batch(iterator)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

        return loss_history

    def _process_batch(self, iterator):
        """Process one accumulated batch.

        Changes applied:
          FIX 1  — output_adjusted shape corrigida para corresponder ao target.
          FIX 5  — saída do modelo já vem com sigmoid (ver summarizer.py);
                   mantemos a MSE pois ambos estão no intervalo [0,1].
          NEW    — ranking_loss adicionada para penalizar ordem errada de frames.
          FIX RL — reward reformulado como diversidade + concordância de ranking,
                   desacoplando-o da MSE supervisionada.
        """
        batch_loss = 0
        # Peso da ranking loss em relação à MSE
        lambda_rank = getattr(self.config, 'lambda_rank', 0.1)
        # Peso do termo de RL em relação à loss supervisionada
        alpha = getattr(self.config, 'alpha_rl', 0.1)

        for _ in range(self.config.batch_size):
            frame_features, target = next(iterator)
            frame_features = frame_features.to(self.config.device)
            target         = target.to(self.config.device)

            output, attn_weights, log_probs, value = self.model(
                frame_features.squeeze(0)
            )

            # ----------------------------------------------------------
            # FIX 1 — shape correta de output_adjusted
            # output vem como [1, T] ou [T] dependendo do squeeze interno;
            # target é [1, T] vindo do DataLoader (batch_size=1).
            # Ambos devem ser [T] antes do MSELoss.
            # ----------------------------------------------------------
            output_adjusted = output.squeeze()       # [T]
            target_squeezed = target.squeeze(0)      # [T]

            # Garante 1D mesmo que T==1
            if output_adjusted.dim() == 0:
                output_adjusted = output_adjusted.unsqueeze(0)
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)

            # ----------------------------------------------------------
            # Loss supervisionada: MSE + Ranking
            # ----------------------------------------------------------
            loss_mse  = nn.MSELoss()(output_adjusted, target_squeezed)

            # FIX NEW — ranking loss: a ordem relativa dos scores importa
            # mais para o F1 do que a magnitude absoluta.
            loss_rank = ranking_loss(output_adjusted, target_squeezed)

            loss_sup = loss_mse + lambda_rank * loss_rank

            # ----------------------------------------------------------
            # FIX RL — reward reformulado
            # Antes: reward = -loss_sup  (circular com a loss supervisionada)
            # Agora: reward baseado na correlação de Spearman aproximada
            # entre scores preditos e targets, desacoplado da MSE.
            # ----------------------------------------------------------
            with torch.no_grad():
                # Correlação de postos aproximada: quanto maior melhor.
                # Usamos a diferença de rankings como proxy.
                pred_rank   = output_adjusted.argsort().float()
                target_rank = target_squeezed.argsort().float()
                rank_corr   = 1.0 - (
                    6.0 * ((pred_rank - target_rank) ** 2).sum()
                    / max((output_adjusted.numel() * (output_adjusted.numel() ** 2 - 1)), 1)
                )
                reward = rank_corr  # [-1, 1] — recompensa o modelo por acertar a ordem

            advantage  = reward - value.squeeze()
            actor_loss = -(log_probs * advantage.detach()).mean()
            critic_loss = advantage.pow(2).mean()
            rl_loss     = actor_loss + critic_loss

            loss = loss_sup + alpha * rl_loss

            loss.backward()
            batch_loss += loss.item()

        return batch_loss / self.config.batch_size

    # ------------------------------------------------------------------
    # Logging & checkpointing
    # ------------------------------------------------------------------

    def _log_epoch_results(self, epoch_i, loss_history):
        """Log results for the current epoch. Returns mean loss."""
        mean_loss = np.mean(loss_history)
        print(f"Epoch {epoch_i} loss: {mean_loss:.4f}")

        if self.config.verbose:
            tqdm.write('Plotting...')

        self.writer.update_loss(mean_loss, epoch_i, 'loss_epoch')
        self._save_checkpoint(epoch_i)

        return mean_loss  # retornado para o scheduler

    def _save_checkpoint(self, epoch_i):
        """Save model checkpoint."""
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch_i}.pkl')
        tqdm.write(f'Saving parameters at {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, epoch_i, save_weights=False):
        """Evaluate the model."""
        self.model.eval()
        out_scores_dict   = {}
        weights_save_path = os.path.join(self.config.score_dir, "weights.h5")

        for frame_features, video_name in tqdm(
            self.test_loader, desc='Evaluate', ncols=80, leave=False
        ):
            scores, attn_weights = self._evaluate_video(frame_features)
            out_scores_dict[video_name] = scores

            if save_weights:
                self._save_attention_weights(
                    weights_save_path, video_name, epoch_i, attn_weights
                )

        self._save_scores(out_scores_dict, epoch_i)

    def _evaluate_video(self, frame_features):
        """Evaluate a single video."""
        frame_features = frame_features.view(-1, self.config.input_size).to(
            self.config.device
        )
        with torch.no_grad():
            scores, attn_weights, _, _ = self.model(frame_features)
            scores      = scores.squeeze(0).cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(self, weights_save_path, video_name, epoch_i, attn_weights):
        """Save attention weights."""
        with h5py.File(weights_save_path, 'a') as weights:
            weights.create_dataset(
                f"{video_name}/epoch_{epoch_i}", data=attn_weights
            )

    def _save_scores(self, out_scores_dict, epoch_i):
        """Save evaluation scores."""
        if not os.path.exists(self.config.score_dir):
            os.makedirs(self.config.score_dir)

        scores_save_path = os.path.join(
            self.config.score_dir,
            f"{self.config.video_type}_{epoch_i}.json",
        )
        with open(scores_save_path, 'w') as f:
            if self.config.verbose:
                tqdm.write(f'Saving scores at {scores_save_path}')
            json.dump(out_scores_dict, f)
        os.chmod(scores_save_path, 0o777)


if __name__ == '__main__':
    pass