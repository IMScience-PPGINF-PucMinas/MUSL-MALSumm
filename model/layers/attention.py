import torch
import torch.nn as nn
from torch.distributions import Bernoulli


class RLAttentionActorCritic(nn.Module):
    def __init__(self, hidden_dim, max_seq_len):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + max_seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        self.actor = nn.Linear(128, max_seq_len)
        self.critic = nn.Linear(128, 1)

    def forward(self, hidden_state, attn_weights):
        B, T, H = hidden_state.shape

        # padding (caso T < max_seq_len)
        if T < self.max_seq_len:
            pad_size = self.max_seq_len - T
            pad = torch.zeros(B, pad_size, device=hidden_state.device)
        
        h_summary = hidden_state.mean(dim=1)              # [B, H]
        attn_summary = attn_weights.mean(dim=1)           # [B, T]

        if T < self.max_seq_len:
            attn_summary = torch.cat([attn_summary, pad], dim=-1)

        state = torch.cat([h_summary, attn_summary], dim=-1)

        features = self.shared(state)

        probs = torch.sigmoid(self.actor(features))       # [B, max_seq_len]

        probs = probs[:, :T]  # corta para tamanho real

        dist = Bernoulli(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)

        value = self.critic(features)

        return actions, log_probs, value

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        # print(f"SEBlock input shape: {x.shape}")
        # print(f"b: {b}, c: {c}")
        y = self.avg_pool(x).view(b, c)
        # print(f"Shape after avg_pool and view: {y.shape}")
        y = self.fc(y).view(b, c, 1)
        # print(f"Shape after fc and view: {y.shape}")
        return x * y.expand_as(x)


class sLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_dim=512, conv_channels=128, dropout=0.5, num_groups=16):
        super(sLSTM, self).__init__()
        self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
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
        
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x = self.ln(x)
        
        out, _ = self.lstm(x)
        
        out = out.permute(0, 2, 1)
        out = self.gn(out)
        out = out.permute(0, 2, 1)
        
        i_gate = torch.exp(self.i_gate(out))
        f_gate = torch.exp(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))
        
        m_gate = torch.max(torch.log(f_gate) + self.m_gate, torch.log(i_gate))
        i_gate_stable = torch.exp(torch.log(i_gate) - m_gate)
        f_gate_stable = torch.exp(torch.log(f_gate) + self.m_gate - m_gate)
        
        z_gate = torch.tanh(self.z_gate(out))
        
        c_gate = f_gate_stable * out + i_gate_stable * z_gate
        out = o_gate * c_gate
        
        out = self.right_linear(out) + self.left_linear(out)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class mLSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_dim=512, num_layers=2, dropout=0.5):
        super(mLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim, hidden_dim)

        self.saoa1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.saoa2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.sig = nn.Sigmoid()
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, input_size)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)

        self.rl_attn = RLAttentionActorCritic(hidden_dim, max_seq_len=200)  # ajuste conforme seu dataset

    
    def forward(self, x):
        out, _ = self.lstm(x)
        
        q = self.q_proj(out)
        k = self.k_proj(out)
        v = self.v_proj(out)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (out.size(-1) ** 0.5)  # Scale attention scores
        attn_weights = torch.softmax(attn_scores, dim=-1)

        actions, log_probs, value = self.rl_attn(out, attn_weights)

        attn_weights = attn_weights * (actions.unsqueeze(1) + 1e-6)

        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        out = out + attn_output
        attn_output = self.sig(self.saoa1(out)) * self.saoa2(out) #adaptive
        out = out + attn_output
        
        i_gate = torch.exp(self.i_gate(out))
        f_gate = torch.exp(self.f_gate(out))
        o_gate = torch.exp(self.o_gate(out))
        gate_sum = i_gate + f_gate + o_gate + 1e-6
        
        i_gate = i_gate / gate_sum
        f_gate = f_gate / gate_sum
        o_gate = o_gate / gate_sum
        
        c_gate = torch.tanh(self.c_gate(out))
        
        out = f_gate * out + i_gate * c_gate
        out = o_gate * out
        
        out = self.layer_norm(out)
        
        out = self.drop(out)
        out = self.fc(out)
        return out, attn_weights, log_probs, value


if __name__ == '__main__':
    pass