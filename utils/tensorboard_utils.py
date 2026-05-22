import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter


class TensorboardWriter(SummaryWriter):
    def __init__(self, logdir: str):
        super().__init__(logdir)
        self.logdir = self.file_writer.get_logdir()

    def update_parameters(self, module: nn.Module, step_i: int) -> None:
        for name, param in module.named_parameters():
            self.add_histogram(name, param.clone().cpu().data.numpy(), step_i)

    def update_loss(self, loss: float, step_i: int, name: str = 'loss') -> None:
        self.add_scalar(name, loss, step_i)

    def update_histogram(self, values, step_i: int, name: str = 'hist') -> None:
        self.add_histogram(name, values, step_i)