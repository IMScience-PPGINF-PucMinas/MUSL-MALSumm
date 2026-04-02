import argparse
import pprint
from pathlib import Path
import torch
from utils.utils import get_paths


def str2bool(v):
    """Convert a string to a boolean value."""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class Config:
    """Configuration class to manage and store all settings."""

    def __init__(self, **kwargs):
        """Initialize configuration with provided keyword arguments."""
        self.log_dir   = None
        self.score_dir = None
        self.save_dir  = None
        self.device    = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        for key, value in kwargs.items():
            setattr(self, key, value)

        self.set_dataset_dir(self.video_type)

    def set_dataset_dir(self, video_type='TVSum'):
        """Set directories for logs, scores, and models based on the dataset."""
        paths = get_paths(video_type)
        self.log_dir   = f"{paths['log_dir']}/split{self.split_index}"
        self.score_dir = f"{paths['score_dir']}/split{self.split_index}"
        self.save_dir  = f"{paths['save_dir']}/split{self.split_index}"

    def __repr__(self):
        """Pretty-print configurations in alphabetical order."""
        config_str  = 'Configurations\n'
        config_str += pprint.pformat(self.__dict__)
        return config_str


def get_config(parse=True, **optional_kwargs):
    """Parse command-line arguments and return a Config object.

    Args:
        parse (bool): Whether to parse command-line arguments.
        optional_kwargs (dict): Additional keyword arguments to override defaults.

    Returns:
        Config: A configuration object with all settings.
    """
    parser = argparse.ArgumentParser()

    # ------------------------------------------------------------------
    # Mode and general settings
    # ------------------------------------------------------------------
    parser.add_argument('--mode', type=str, default='train',
                        help='Mode for the configuration [train | test]')
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='Print training messages')
    parser.add_argument('--video_type', type=str, default='SumMe',
                        help='Dataset to be used [SumMe | TVSum | MrHiSum | both]')

    # ------------------------------------------------------------------
    # Model settings
    # ------------------------------------------------------------------
    parser.add_argument('--input_size', type=int, default=1024,
                        help='Feature size expected in the input')
    parser.add_argument('--seed', type=int, default=12345,
                        help='Random seed for reproducibility')
    parser.add_argument('--fusion', type=str, default='add',
                        help='Type of feature fusion')
    parser.add_argument('--n_segments', type=int, default=4,
                        help='Number of segments to split the video')

    # FIX ARCH — positional encoding está definido nos configs mas não era
    # aplicado no forward(). O argumento é mantido para que summarizer.py
    # possa consumi-lo após a correção naquele arquivo.
    parser.add_argument('--pos_enc', type=str, default='absolute',
                        help='Type of positional encoding [absolute | relative | none]')
    parser.add_argument('--heads', type=int, default=8,
                        help='Number of global heads for the attention module')

    # ------------------------------------------------------------------
    # Training settings
    # ------------------------------------------------------------------
    parser.add_argument('--n_epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('--clip', type=float, default=5.0,
                        help='Max norm of the gradients')
    parser.add_argument('--lr', type=float, default=5e-5,
                        help='Learning rate')
    parser.add_argument('--l2_req', type=float, default=1e-5,
                        help='L2 regularization factor')
    parser.add_argument('--split_index', type=int, default=0,
                        help='Data split to be used [0-4]')
    parser.add_argument('--init_type', type=str, default='xavier',
                        help='Weight initialization [xavier | kaiming | normal | orthogonal]')
    parser.add_argument('--init_gain', type=float, default=None,
                        help='Scaling factor for weight initialization')

    # ------------------------------------------------------------------
    # LR Scheduler settings  (NEW)
    # ReduceLROnPlateau reduz o lr quando a loss estagna, evitando
    # oscilações nas épocas finais com lr fixo.
    # ------------------------------------------------------------------
    parser.add_argument('--lr_scheduler_patience', type=int, default=10,
                        help='Epochs without improvement before reducing lr')
    parser.add_argument('--lr_scheduler_factor', type=float, default=0.5,
                        help='Factor by which lr is reduced on plateau')

    # ------------------------------------------------------------------
    # Loss weights  (NEW)
    # Permitem ajustar a contribuição de cada termo da loss sem mexer
    # no código do solver.
    # ------------------------------------------------------------------
    parser.add_argument('--lambda_rank', type=float, default=0.1,
                        help='Weight of the pairwise ranking loss term '
                             '(0 = desativado, recomendado: 0.05-0.2)')
    parser.add_argument('--alpha_rl', type=float, default=0.1,
                        help='Weight of the RL (actor-critic) loss term '
                             '(0 = desativado, recomendado: 0.05-0.15)')

    # ------------------------------------------------------------------
    # Splits  (NEW)
    # Controla se os splits aumentados (_aug) devem ser usados.
    # O data_loader detecta automaticamente o arquivo; esta flag permite
    # forçar o uso do split padrão quando necessário (ex: ablation study).
    # ------------------------------------------------------------------
    parser.add_argument('--use_aug_splits', type=str2bool, default=True,
                        help='Use augmented splits (*_aug_splits.json) when '
                             'available — typically improves F1 by ~1-2%%')

    # ------------------------------------------------------------------
    # LSTM-specific settings
    # ------------------------------------------------------------------
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension of LSTM')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout rate for LSTM')

    # ------------------------------------------------------------------
    # RL / attention settings  (NEW)
    # max_seq_len estava hardcoded como 200 dentro de attention.py,
    # causando crash silencioso em vídeos com T > 200 frames.
    # Agora é configurável e passado explicitamente ao modelo.
    # ------------------------------------------------------------------
    parser.add_argument('--max_seq_len', type=int, default=200,
                        help='Maximum sequence length for RLAttentionActorCritic. '
                             'Aumentar para datasets com vídeos longos (ex: 500).')

    if parse:
        kwargs = parser.parse_args()
    else:
        kwargs = parser.parse_known_args()[0]

    kwargs = vars(kwargs)
    kwargs.update(optional_kwargs)

    return Config(**kwargs)


if __name__ == '__main__':
    config = get_config()
    print(config)