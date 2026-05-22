import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import sLSTM, mLSTM, SEBlock


class xLSTM(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_segments,
        hidden_dim=512,
        num_layers=2,
        dropout=0.5,
        lstm_type='mLSTM'
    ):
        super(xLSTM, self).__init__()

        # ---------------------------------------------------
        # Residual projection
        # ---------------------------------------------------
        self.input_proj = nn.Conv1d(
            input_size,
            hidden_dim,
            kernel_size=1
        )

        # ---------------------------------------------------
        # Parallel recurrent branches
        # ---------------------------------------------------
        self.slstm = sLSTM(
            input_size,
            hidden_dim,
            dropout=dropout
        )

        self.mlstm = mLSTM(
            input_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout
        )

        # ---------------------------------------------------
        # Gated fusion
        # ---------------------------------------------------
        self.attn_linear = nn.Linear(
            hidden_dim * 2,
            hidden_dim
        )

        # ---------------------------------------------------
        # Channel attention
        # ---------------------------------------------------
        self.se_block = SEBlock(hidden_dim)

        # ---------------------------------------------------
        # Temporal refinement
        # Depthwise temporal convolution
        # ---------------------------------------------------
        self.temporal_refine = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )

        # ---------------------------------------------------
        # Normalization
        # ---------------------------------------------------
        self.norm = nn.LayerNorm(hidden_dim)

        # ---------------------------------------------------
        # Prediction head
        # ---------------------------------------------------
        self.fc = nn.Linear(
            hidden_dim,
            output_size
        )

        self.fc_output = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size // 2, 1)
        )

        self.num_segments = num_segments

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())

        trainable = sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad
        )

        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def forward(self, x):
        """
        Input:
            x -> (B, T, input_size) ou (T, input_size)
        """
        # ---------------------------------------------------
        # Garante dimensão de batch
        # ---------------------------------------------------
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(0)   # (T, C) -> (1, T, C)
            squeeze_output = True

        # ---------------------------------------------------
        # Residual projection
        # Conv1d expects (B, C, T)
        # ---------------------------------------------------
        residual = self.input_proj(
            x.permute(0, 2, 1)
        ).permute(0, 2, 1)

        # ---------------------------------------------------
        # Parallel recurrent branches
        # ---------------------------------------------------
        x_slstm = self.slstm(x)

        x_mlstm, attn_weights = self.mlstm(x)

        # ---------------------------------------------------
        # Gated fusion
        # ---------------------------------------------------
        fusion_input = torch.cat(
            [x_slstm, x_mlstm],
            dim=-1
        )

        gate = torch.sigmoid(
            self.attn_linear(fusion_input)
        )

        x_combined = (
            gate * x_slstm +
            (1.0 - gate) * x_mlstm
        )

        # ---------------------------------------------------
        # Global residual connection
        # ---------------------------------------------------
        x_combined = x_combined + residual

        # ---------------------------------------------------
        # SE channel refinement
        # ---------------------------------------------------
        x_se = self.se_block(
            x_combined.permute(0, 2, 1)
        ).permute(0, 2, 1)

        x_combined = x_combined + x_se

        # ---------------------------------------------------
        # Temporal refinement
        # ---------------------------------------------------
        x_ref = self.temporal_refine(
            x_combined.permute(0, 2, 1)
        ).permute(0, 2, 1)

        x_combined = x_combined + x_ref

        # ---------------------------------------------------
        # Final normalization
        # ---------------------------------------------------
        x_combined = self.norm(x_combined)

        # ---------------------------------------------------
        # Prediction head
        # ---------------------------------------------------
        output = self.fc(x_combined)
        output = self.fc_output(output)

        # (B, T, 1) -> (B, T)
        output = output.squeeze(-1)

        # ---------------------------------------------------
        # Remove dimensão de batch se foi adicionada aqui
        # ---------------------------------------------------
        if squeeze_output:
            output = output.squeeze(0)        # (1, T) -> (T,)
            # attn_weights pode ser None ou tensor — trata os dois casos
            if attn_weights is not None and attn_weights.dim() == 3:
                attn_weights = attn_weights.squeeze(0)

        return output, attn_weights