import torch
import torch.nn as nn

from .attention import sLSTM, mLSTM, SEBlock


class xLSTM(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_segments: int,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.num_segments = num_segments

        self.input_proj = nn.Conv1d(input_size, hidden_dim, kernel_size=1)
        self.slstm = sLSTM(input_size, hidden_dim, dropout=dropout)
        self.mlstm = mLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)
        self.attn_linear = nn.Linear(input_size * 2, input_size)
        self.fusion_proj = nn.Linear(input_size, hidden_dim)
        self.se_block = SEBlock(hidden_dim)
        self.temporal_refine = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_size)
        self.fc_output = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size // 2, 1),
        )

    def count_parameters(self) -> None:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def forward(self, x: torch.Tensor):
        squeeze_output = x.dim() == 2
        if squeeze_output:
            x = x.unsqueeze(0)

        residual = self.input_proj(x.permute(0, 2, 1)).permute(0, 2, 1)

        x_slstm = self.slstm(x)
        x_mlstm, attn_weights = self.mlstm(x)

        gate = torch.sigmoid(self.attn_linear(torch.cat([x_slstm, x_mlstm], dim=-1)))
        x_combined = gate * x_slstm + (1.0 - gate) * x_mlstm

        x_combined = self.fusion_proj(x_combined) + residual

        x_ref = self.temporal_refine(x_combined.permute(0, 2, 1)).permute(0, 2, 1)
        x_combined = self.norm(x_combined + x_ref)

        output = self.fc_output(self.fc(x_combined)).squeeze(-1)

        if squeeze_output:
            output = output.squeeze(0)
            if attn_weights is not None and attn_weights.dim() == 3:
                attn_weights = attn_weights.squeeze(0)

        return output, attn_weights