import torch
import torch.nn as nn

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
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class sLSTM(nn.Module):
    def __init__(
        self,
        input_size=1024,
        hidden_dim=512,
        conv_channels=128,
        dropout=0.5,
        num_groups=16,
        max_len=1024,
        num_heads=8,
    ):
        super(sLSTM, self).__init__()
 
        self.input_proj = nn.Conv1d(input_size, conv_channels, kernel_size=1)
        self.conv3 = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(input_size, conv_channels, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(input_size, conv_channels, kernel_size=7, padding=3)
        self.conv_fusion = nn.Conv1d(conv_channels * 3, conv_channels, kernel_size=1)
 
        self.max_len = max_len
        self.pos_emb = nn.Embedding(max_len, conv_channels)
 
        self.ln = nn.LayerNorm(conv_channels)
        self.lstm = nn.LSTM(
            conv_channels, hidden_dim, num_layers=2, batch_first=True, dropout=dropout
        )
        self.gn = nn.GroupNorm(num_groups, hidden_dim)
 
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(hidden_dim)
 
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.z_gate = nn.Linear(hidden_dim, hidden_dim)
        self.m_gate = nn.Parameter(torch.zeros(1))
 
        self.right_linear = nn.Linear(hidden_dim, hidden_dim)
        self.left_linear  = nn.Linear(hidden_dim, hidden_dim)
        self.proj         = nn.Linear(hidden_dim, input_size)
        self.dropout      = nn.Dropout1d(dropout)
 
        self.fc_gate = nn.Linear(input_size, input_size)
        self.fc_proj = nn.Linear(input_size, input_size)
 
        self.res_proj = nn.Sequential(
            nn.Linear(conv_channels, hidden_dim),
            nn.GELU()
        )
 
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
 
        x = x.permute(0, 2, 1)          # (B, input_size, T) → conv espera (B, C, T)
        residual = self.input_proj(x) 
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        x = torch.cat([c3, c5, c7], dim=1)
        x = self.conv_fusion(x)          # (B, conv_channels, T)
        x = x + residual 
        x = x.permute(0, 2, 1)          # (B, T, conv_channels)
 
        T = x.size(1)
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # (1, T)
        x = x + self.pos_emb(positions)  # broadcast sobre o batch
 
        x = self.ln(x)
 
        res = self.res_proj(x)           # (B, T, hidden_dim)
        out, _ = self.lstm(x)            # (B, T, hidden_dim)
        out = out + res
 
        out = out.permute(0, 2, 1)
        out = self.gn(out)
        out = out.permute(0, 2, 1)       # (B, T, hidden_dim)
 
        attn_out, _ = self.temporal_attn(out, out, out)
        out = self.attn_norm(out + attn_out)   # residual + layer norm
 
        i_gate = torch.sigmoid(self.i_gate(out))
        f_gate = torch.sigmoid(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))
 
        m_gate = torch.max(
            torch.log(f_gate) + self.m_gate,
            torch.log(i_gate)
        )
        i_gate_stable = torch.exp(torch.log(i_gate) - m_gate)
        f_gate_stable = torch.exp(torch.log(f_gate) + self.m_gate - m_gate)
 
        z_gate = torch.tanh(self.z_gate(out))
        c_gate = f_gate_stable * out + i_gate_stable * z_gate
        out    = o_gate * c_gate
 
        out = self.right_linear(out) + self.left_linear(out)
        out = self.proj(out)             # (B, T, input_size)
 
        out = out.permute(0, 2, 1)
        out = self.dropout(out)
        out = out.permute(0, 2, 1)       # (B, T, input_size)
 
        gate = torch.sigmoid(self.fc_gate(out))
        out  = gate * self.fc_proj(out) + (1.0 - gate) * out
 
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
        
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, input_size)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        
        q = self.q_proj(out)
        k = self.k_proj(out)
        v = self.v_proj(out)
        
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (out.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        
        attn_output = torch.matmul(attn_weights, v)
        
        out = out + attn_output
        
        i_gate = torch.sigmoid(self.i_gate(out))
        f_gate = torch.sigmoid(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))
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
        return out, attn_weights


if __name__ == '__main__':
    pass
