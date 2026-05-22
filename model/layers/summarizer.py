import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import sLSTM, mLSTM, SEBlock

class xLSTM(nn.Module):
    def __init__(self, input_size, output_size, num_segments, hidden_dim=512, num_layers=2, dropout=0.5, lstm_type='mLSTM'):
        super(xLSTM, self).__init__()
        self.input_proj = nn.Conv1d(input_size, 128, kernel_size=1)
        self.slstm = sLSTM(input_size, hidden_dim, dropout=dropout)
        self.mlstm = mLSTM(input_size, hidden_dim, num_layers=num_layers, dropout=dropout)

        self.se_block = SEBlock(input_size)
        self.conv = nn.Conv1d(input_size, hidden_dim, kernel_size=1)
        
        self.attn_linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.attn_softmax = nn.Softmax(dim=-1)
        
        self.fc = nn.Linear(input_size, output_size)
        self.fc_output = nn.Sequential(
            nn.Linear(output_size, output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_size // 2, 1)
        )
        self.num_segments = num_segments

        self.temporal_refine = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            padding=1,
            groups=hidden_dim
        )

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    def forward(self, x):
        residual = self.input_proj(x)

        x_slstm = self.slstm(x)
        x_mlstm, attn_weights = self.mlstm(x)

        fusion_input = torch.cat([x_slstm, x_mlstm], dim=-1)
        gate = torch.sigmoid(self.attn_linear(fusion_input))

        x_combined = gate * x_slstm + (1 - gate) * x_mlstm

        x_combined = x_combined + residual

        x_combined = self.norm(x_combined)
        
        x_se = self.se_block(x_combined.permute(0,2,1)).permute(0,2,1)

        x_combined = x_combined + x_se

        x_ref = x_combined.permute(0,2,1)
        x_ref = self.temporal_refine(x_ref)
        x_ref = x_ref.permute(0,2,1)

        x_combined = x_combined + x_ref
        
        output = self.fc(x_combined)
        output = self.fc_output(output)
        output = output.view(output.size(0), -1)
        
        return output, attn_weights


if __name__ == '__main__':
    pass
