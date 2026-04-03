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

        # --- Early stopping state ---
        # Melhor F1 médio visto até agora (calculado a partir dos scores JSON)
        self.best_f1               = -1.0
        # Época em que o melhor F1 foi atingido
        self.best_epoch            = -1
        # Contador de épocas sem melhora no F1
        self.epochs_no_improve     = 0

        self._set_random_seed()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _set_random_seed(self):
        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)
            np.random.seed(self.config.seed)
            random.seed(self.config.seed)

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
            dropout=self.config.dropout,
            max_seq_len=getattr(self.config, 'max_seq_len', 500),
            pos_enc=getattr(self.config, 'pos_enc', 'absolute'),
        ).to(self.config.device)

        if self.config.init_type is not None:
            self.init_weights(
                self.model,
                init_type=self.config.init_type,
                init_gain=self.config.init_gain,
            )

    def _initialize_optimizer_and_writer(self):
        if self.config.mode == 'train':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.lr,
                weight_decay=self.config.l2_req,
            )
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=getattr(self.config, 'lr_scheduler_factor', 0.5),
                patience=getattr(self.config, 'lr_scheduler_patience', 10),
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
                    raise NotImplementedError(
                        f"Initialization method {init_type} is not implemented."
                    )
            elif 'bias' in name or param.dim() < 2:
                nn.init.constant_(param, 0.1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        """Train the model with F1-based early stopping.

        A cada época:
          1. treina um epoch
          2. avalia e salva os scores JSON
          3. estima o F1 médio a partir dos scores salvos
          4. se melhorou → salva 'best_model.pkl' e zera contador
          5. se não melhorou por `patience` épocas → para o treino
        """
        patience = getattr(self.config, 'early_stop_patience', 30)

        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            self.model.train()
            loss_history = self._train_one_epoch(epoch_i)
            mean_loss    = self._log_epoch_results(epoch_i, loss_history)

            if self.scheduler is not None:
                self.scheduler.step(mean_loss)

            self.evaluate(epoch_i)

            # ----------------------------------------------------------
            # Early stopping baseado em F1
            # Lê os scores JSON recém-salvos e calcula um proxy de F1:
            # a média dos scores de cada vídeo correlaciona com o F1 real,
            # mas o cálculo exato requer generate_summary + evaluate_summary.
            # Usamos a variância dos scores como proxy de qualidade:
            # scores colapsados (todos iguais) → variância ≈ 0 → F1 baixo.
            # Quando disponível, usa o F1 calculado externamente via
            # config.score_dir/f1_epoch_{epoch_i}.txt (gerado pelo
            # script de avaliação se rodado em paralelo).
            # ----------------------------------------------------------
            current_f1 = self._read_epoch_f1(epoch_i)

            if current_f1 > self.best_f1:
                self.best_f1    = current_f1
                self.best_epoch = epoch_i
                self.epochs_no_improve = 0
                self._save_best_checkpoint()
                tqdm.write(
                    f"  → Novo melhor F1 proxy: {current_f1:.4f} "
                    f"(época {epoch_i}) — best_model.pkl atualizado"
                )
            else:
                self.epochs_no_improve += 1
                tqdm.write(
                    f"  Sem melhora há {self.epochs_no_improve} épocas "
                    f"(melhor: {self.best_f1:.4f} na época {self.best_epoch})"
                )
                if self.epochs_no_improve >= patience:
                    tqdm.write(
                        f"\nEarly stopping na época {epoch_i}. "
                        f"Melhor F1 proxy: {self.best_f1:.4f} "
                        f"(época {self.best_epoch})"
                    )
                    break

        tqdm.write(
            f"\nTreino concluído. Melhor época: {self.best_epoch} "
            f"| F1 proxy: {self.best_f1:.4f}"
        )

    def _read_epoch_f1(self, epoch_i):
        """Estima o F1 da época a partir dos scores JSON salvos.

        Estratégia em camadas:
          1. Se existir um arquivo f1_epoch_{epoch_i}.txt no score_dir
             (gerado por script externo), usa o valor real.
          2. Caso contrário, usa a variância média dos scores como proxy:
             - variância alta → scores discriminativos → F1 tende a ser maior
             - variância ≈ 0  → modelo colapsou → F1 baixo

        Este proxy não substitui o F1 real, mas é suficiente para detectar
        colapso e melhora consistente sem precisar rodar generate_summary
        a cada época dentro do solver.
        """
        # Tentativa 1: arquivo de F1 real gerado externamente
        f1_file = os.path.join(
            self.config.score_dir, f"f1_epoch_{epoch_i}.txt"
        )
        if os.path.exists(f1_file):
            try:
                with open(f1_file) as fp:
                    return float(fp.read().strip())
            except (ValueError, IOError):
                pass

        # Tentativa 2: proxy por variância dos scores
        scores_path = os.path.join(
            self.config.score_dir,
            f"{self.config.video_type}_{epoch_i}.json",
        )
        if not os.path.exists(scores_path):
            return -1.0

        try:
            with open(scores_path) as fp:
                scores_dict = json.load(fp)

            variances = []
            for video_scores in scores_dict.values():
                arr = np.array(video_scores, dtype=float)
                if arr.size > 1:
                    variances.append(float(np.var(arr)))

            if not variances:
                return -1.0

            # Normaliza pela variância máxima esperada (0.25 para sigmoid)
            return float(np.mean(variances)) / 0.25

        except (json.JSONDecodeError, Exception):
            return -1.0

    def _train_one_epoch(self, epoch_i):
        loss_history = []
        num_batches  = len(self.train_loader) // self.config.batch_size
        iterator     = iter(self.train_loader)

        for _ in trange(num_batches, desc='Batch', ncols=80, leave=False):
            self.optimizer.zero_grad()
            batch_loss = self._process_batch(iterator, epoch_i)
            loss_history.append(batch_loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
            self.optimizer.step()

        return loss_history

    def _process_batch(self, iterator, epoch_i=0):
        """Processa um batch acumulado.

        Mudanças nesta versão:
          - alpha_rl com decay exponencial por época: o RL contribui mais
            no início (exploração) e menos no final (estabilização).
            Isso evita o colapso observado após a época 160 no SumMe,
            onde o actor-critic dominava o sinal supervisionado.
          - lambda_rank e alpha_rl lidos do config com fallback seguro.
        """
        batch_loss  = 0
        lambda_rank = getattr(self.config, 'lambda_rank', 0.1)

        # --- RL decay exponencial ---
        # alpha_rl decai de alpha_rl_start até alpha_rl_end ao longo de
        # n_epochs épocas. Com os defaults: 0.1 → 0.01 em 200 épocas.
        alpha_start = getattr(self.config, 'alpha_rl', 0.1)
        alpha_end   = getattr(self.config, 'alpha_rl_end', 0.01)
        n_epochs    = max(self.config.n_epochs - 1, 1)
        # Interpolação exponencial entre alpha_start e alpha_end
        decay_ratio = (alpha_end / max(alpha_start, 1e-8)) ** (epoch_i / n_epochs)
        alpha       = alpha_start * decay_ratio

        for _ in range(self.config.batch_size):
            frame_features, target = next(iterator)
            frame_features = frame_features.to(self.config.device)
            target         = target.to(self.config.device)

            output, attn_weights, log_probs, value = self.model(
                frame_features.squeeze(0)
            )

            output_adjusted = output.squeeze()
            target_squeezed = target.squeeze(0)

            if output_adjusted.dim() == 0:
                output_adjusted = output_adjusted.unsqueeze(0)
            if target_squeezed.dim() == 0:
                target_squeezed = target_squeezed.unsqueeze(0)

            # Loss supervisionada: MSE + Ranking
            loss_mse  = nn.MSELoss()(output_adjusted, target_squeezed)
            loss_rank = ranking_loss(output_adjusted, target_squeezed)
            loss_sup  = loss_mse + lambda_rank * loss_rank

            # RL com reward de correlação de ranking (Spearman aproximado)
            with torch.no_grad():
                pred_rank   = output_adjusted.argsort().float()
                target_rank = target_squeezed.argsort().float()
                n           = output_adjusted.numel()
                denom       = max(n * (n ** 2 - 1), 1)
                rank_corr   = 1.0 - 6.0 * ((pred_rank - target_rank) ** 2).sum() / denom
                reward      = rank_corr

            advantage   = reward - value.squeeze()
            actor_loss  = -(log_probs * advantage.detach()).mean()
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
        mean_loss = np.mean(loss_history)
        print(f"Epoch {epoch_i} loss: {mean_loss:.4f}")

        if self.config.verbose:
            tqdm.write('Plotting...')

        self.writer.update_loss(mean_loss, epoch_i, 'loss_epoch')
        self._save_checkpoint(epoch_i)
        return mean_loss

    def _save_checkpoint(self, epoch_i):
        """Salva checkpoint da época atual."""
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        ckpt_path = os.path.join(self.config.save_dir, f'epoch-{epoch_i}.pkl')
        tqdm.write(f'Saving parameters at {ckpt_path}')
        torch.save(self.model.state_dict(), ckpt_path)

    def _save_best_checkpoint(self):
        """Salva o melhor modelo como 'best_model.pkl'.

        Mantido separado dos checkpoints por época para não ser sobrescrito
        nas épocas seguintes e para facilitar o uso no inference.py.
        """
        if not os.path.exists(self.config.save_dir):
            os.makedirs(self.config.save_dir)
        best_path = os.path.join(self.config.save_dir, 'best_model.pkl')
        torch.save(self.model.state_dict(), best_path)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, epoch_i, save_weights=False):
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
        frame_features = frame_features.view(-1, self.config.input_size).to(
            self.config.device
        )
        with torch.no_grad():
            scores, attn_weights, _, _ = self.model(frame_features)
            scores       = scores.squeeze(0).cpu().numpy().tolist()
            attn_weights = attn_weights.cpu().numpy()
        return scores, attn_weights

    def _save_attention_weights(self, weights_save_path, video_name, epoch_i, attn_weights):
        with h5py.File(weights_save_path, 'a') as weights:
            weights.create_dataset(
                f"{video_name}/epoch_{epoch_i}", data=attn_weights
            )

    def _save_scores(self, out_scores_dict, epoch_i):
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
