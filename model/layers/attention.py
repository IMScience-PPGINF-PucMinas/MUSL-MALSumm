import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channel: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class sLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_dim: int = 512,
        conv_channels: int = 128,
        dropout: float = 0.5,
        num_groups: int = 16,
    ):
        super().__init__()
        self.conv3 = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_size, conv_channels, kernel_size=7, padding=3)
        self.conv_fusion = nn.Conv1d(conv_channels * 3, conv_channels, kernel_size=1)
        self.ln = nn.LayerNorm(conv_channels)
        self.lstm = nn.LSTM(conv_channels, hidden_dim, num_layers=1, batch_first=True, dropout=dropout)
        self.gn = nn.GroupNorm(num_groups, hidden_dim)
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.z_gate = nn.Linear(hidden_dim, hidden_dim)
        self.m_gate = nn.Parameter(torch.zeros(1))
        self.right_linear = nn.Linear(hidden_dim, hidden_dim)
        self.left_linear = nn.Linear(hidden_dim, hidden_dim)
        self.proj = nn.Linear(hidden_dim, input_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(input_size, input_size)
        self.res_proj = nn.Sequential(nn.Linear(conv_channels, hidden_dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)

        x = x.permute(0, 2, 1)
        fused = torch.cat([self.conv3(x), self.conv5(x), self.conv7(x)], dim=1)
        x = self.conv_fusion(fused).permute(0, 2, 1)
        x = self.ln(x)

        res = self.res_proj(x)
        out, _ = self.lstm(x)
        out = out + res

        out = self.gn(out.permute(0, 2, 1)).permute(0, 2, 1)

        i = torch.exp(self.i_gate(out))
        f = torch.exp(self.f_gate(out))
        o = torch.sigmoid(self.o_gate(out))

        m = torch.max(torch.log(f) + self.m_gate, torch.log(i))
        i_stable = torch.exp(torch.log(i) - m)
        f_stable = torch.exp(torch.log(f) + self.m_gate - m)

        z = torch.tanh(self.z_gate(out))
        c = f_stable * out + i_stable * z
        out = o * c

        out = self.right_linear(out) + self.left_linear(out)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class mLSTM(nn.Module):
    def __init__(
        self,
        input_size: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, input_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        x_ext: torch.Tensor | None = None,
        ext_k: nn.Linear | None = None,
        ext_v: nn.Linear | None = None,
    ):
        out, _ = self.lstm(x)

        q = self.q_proj(out)
        if x_ext is not None and ext_k is not None and ext_v is not None:
            k = ext_k(x_ext)
            v = ext_v(x_ext)
        else:
            k = self.k_proj(out)
            v = self.v_proj(out)

        scale = out.size(-1) ** 0.5
        attn_weights = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
        out = out + torch.matmul(attn_weights, v)

        i = torch.exp(self.i_gate(out))
        f = torch.exp(self.f_gate(out))
        o = torch.exp(self.o_gate(out))
        gate_sum = i + f + o + 1e-6
        i, f, o = i / gate_sum, f / gate_sum, o / gate_sum

        c = torch.tanh(self.c_gate(out))
        out = o * (f * out + i * c)
        out = self.layer_norm(out)
        out = self.fc(self.drop(out))

        return out, attn_weights