import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli


class RLAttentionActorCritic(nn.Module):
    """Actor-Critic que modula os pesos de atenção com ações binárias por frame.

    O actor aprende uma máscara de seleção sobre os T frames; o critic
    estima o valor do estado atual para calcular a vantagem.

    FIX — max_seq_len agora é recebido como parâmetro (era hardcoded 200),
    permitindo ajuste via configs.py sem alterar o código do modelo.
    O padding/truncamento foi corrigido para cobrir T >= max_seq_len.
    """

    def __init__(self, hidden_dim, max_seq_len=200):
        super().__init__()
        self.max_seq_len = max_seq_len

        self.shared = nn.Sequential(
            nn.Linear(hidden_dim + max_seq_len, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(128, max_seq_len)
        self.critic = nn.Linear(128, 1)

    def forward(self, hidden_state, attn_weights):
        """
        Args:
            hidden_state: [B, T, H]
            attn_weights: [B, T, T]
        Returns:
            actions:   [B, T]  — ações binárias amostradas
            log_probs: [B, T]  — log-probabilidades das ações
            value:     [B, 1]  — estimativa do crítico
        """
        B, T, H = hidden_state.shape

        h_summary   = hidden_state.mean(dim=1)       # [B, H]
        attn_summary = attn_weights.mean(dim=1)       # [B, T]

        # FIX — tratamento correto de T vs max_seq_len em ambos os sentidos:
        # · T < max_seq_len → pad com zeros à direita
        # · T >= max_seq_len → trunca para max_seq_len
        # O código original criava o pad mas só o concatenava quando T < max_seq_len,
        # deixando attn_summary com shape [B, T] quando T >= max_seq_len e causando
        # erro de dimensão no Linear (esperava hidden_dim + max_seq_len).
        if T < self.max_seq_len:
            pad = torch.zeros(B, self.max_seq_len - T, device=hidden_state.device)
            attn_summary = torch.cat([attn_summary, pad], dim=-1)  # [B, max_seq_len]
        elif T > self.max_seq_len:
            attn_summary = attn_summary[:, :self.max_seq_len]       # [B, max_seq_len]
        # T == max_seq_len → sem alteração

        state    = torch.cat([h_summary, attn_summary], dim=-1)    # [B, H+max_seq_len]
        features = self.shared(state)                               # [B, 128]

        probs = torch.sigmoid(self.actor(features))                 # [B, max_seq_len]
        probs = probs[:, :T]                                        # [B, T] — recorta para T real

        dist      = Bernoulli(probs)
        actions   = dist.sample()
        log_probs = dist.log_prob(actions)

        value = self.critic(features)                               # [B, 1]

        return actions, log_probs, value


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block para recalibração de canais.

    Recalibra globalmente a importância de cada dimensão de feature,
    funcionando como atenção sobre o espaço de features (não temporal).
    """

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: [B, C, T]
        Returns:
            x recalibrado: [B, C, T]
        """
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)      # [B, C]
        y = self.fc(y).view(b, c, 1)         # [B, C, 1]
        return x * y.expand_as(x)


class sLSTM(nn.Module):
    """LSTM escalar com controle de memória por gates estabilizados.

    Pipeline:
      Conv1d (extração local) → LayerNorm → LSTM → GroupNorm
      → gates i/f/o estabilizados em log-espaço
      → célula de memória: c = f*out + i*z
      → fusão bidirecional (right + left) → projeção → dropout → fc

    O gate de forget (f) pondera a memória recorrente existente (out) e o
    gate de input (i) injeta novo conteúdo (z). A estabilização numérica
    via m_gate evita overflow de exp em sequências longas.

    Nota: nn.LSTM com num_layers=1 ignora o parâmetro dropout internamente
    (PyTorch só aplica dropout entre camadas). O dropout explícito no final
    do forward cobre esse caso.
    """

    def __init__(self, input_size=1024, hidden_dim=512, conv_channels=128,
                 dropout=0.5, num_groups=16):
        super(sLSTM, self).__init__()

        self.conv = nn.Conv1d(input_size, conv_channels, kernel_size=3, padding=1)
        self.ln   = nn.LayerNorm(conv_channels)

        # dropout ignorado pelo PyTorch com num_layers=1; mantido para
        # compatibilidade caso num_layers seja aumentado futuramente.
        self.lstm = nn.LSTM(conv_channels, hidden_dim, num_layers=1,
                            batch_first=True, dropout=dropout)
        self.gn   = nn.GroupNorm(num_groups, hidden_dim)

        # Gates de memória — usam exp → estabilização em log-espaço via m_gate
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.z_gate = nn.Linear(hidden_dim, hidden_dim)

        # Escalar aprendível para estabilização numérica do forget gate
        self.m_gate = nn.Parameter(torch.zeros(1))

        self.right_linear = nn.Linear(hidden_dim, hidden_dim)
        self.left_linear  = nn.Linear(hidden_dim, hidden_dim)
        self.proj         = nn.Linear(hidden_dim, input_size)
        self.dropout      = nn.Dropout(dropout)
        self.fc           = nn.Linear(input_size, input_size)

    def forward(self, x):
        """
        Args:
            x: [T, F] ou [B, T, F]
        Returns:
            out: [B, T, input_size]
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)                  # [1, T, F]

        # Extração local via Conv1d
        x = x.permute(0, 2, 1)                  # [B, F, T]
        x = self.conv(x)                         # [B, conv_channels, T]
        x = x.permute(0, 2, 1)                  # [B, T, conv_channels]
        x = self.ln(x)

        out, _ = self.lstm(x)                    # [B, T, hidden_dim]

        # Normalização de grupo sobre a dimensão temporal
        out = out.permute(0, 2, 1)               # [B, hidden_dim, T]
        out = self.gn(out)
        out = out.permute(0, 2, 1)               # [B, T, hidden_dim]

        # Gates com estabilização numérica em log-espaço
        i_gate = torch.exp(self.i_gate(out))
        f_gate = torch.exp(self.f_gate(out))
        o_gate = torch.sigmoid(self.o_gate(out))

        # m_gate: máximo entre log(f) + m e log(i) para estabilizar exp
        m_gate = torch.max(
            torch.log(f_gate) + self.m_gate,
            torch.log(i_gate),
        )
        i_gate_stable = torch.exp(torch.log(i_gate) - m_gate)
        f_gate_stable = torch.exp(torch.log(f_gate) + self.m_gate - m_gate)

        z_gate = torch.tanh(self.z_gate(out))

        # Célula de memória: balanço entre memória existente (f) e novo conteúdo (i)
        c_gate = f_gate_stable * out + i_gate_stable * z_gate
        out    = o_gate * c_gate

        # Fusão bidirecional + projeção de volta para input_size
        out = self.right_linear(out) + self.left_linear(out)
        out = self.proj(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class mLSTM(nn.Module):
    """LSTM matricial com atenção Q/K/V e modulação por RL Actor-Critic.

    Pipeline:
      LSTM multi-camada → Q/K/V attention → modulação RL (máscara de frames)
      → gate adaptativo SAOA → gates i/f/o normalizados (balanço de memória)
      → LayerNorm → dropout → projeção final

    O mecanismo de gates normalizado (i+f+o com divisão pela soma) implementa
    competição entre os três componentes de memória — uma forma de softmax
    sobre quanto de cada gate contribui para o estado final. Isso é
    intencional para o controle de quantidade de memória utilizada.

    SAOA (Self-Attentive Output Activation): sigmoid(W1·out) * W2·out
    é um gate multiplicativo adaptativo que filtra a saída da atenção.
    """

    def __init__(self, input_size=1024, hidden_dim=512, num_layers=2,
                 dropout=0.5, max_seq_len=200):
        super(mLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout)

        # Projeções Q/K/V para atenção sobre saídas do LSTM
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # Gates de memória com balanço normalizado
        self.i_gate = nn.Linear(hidden_dim, hidden_dim)
        self.f_gate = nn.Linear(hidden_dim, hidden_dim)
        self.o_gate = nn.Linear(hidden_dim, hidden_dim)
        self.c_gate = nn.Linear(hidden_dim, hidden_dim)

        # Gate adaptativo SAOA
        self.saoa1 = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.saoa2 = nn.Linear(hidden_dim, hidden_dim, bias=False)

        self.sig        = nn.Sigmoid()
        self.drop       = nn.Dropout(dropout)
        self.fc         = nn.Linear(hidden_dim, input_size)
        self.layer_norm = nn.LayerNorm(hidden_dim)

        # FIX — max_seq_len recebido como parâmetro em vez de hardcoded 200.
        # Deve ser passado pelo xLSTM a partir de config.max_seq_len.
        self.rl_attn = RLAttentionActorCritic(
            hidden_dim=hidden_dim,
            max_seq_len=max_seq_len,
        )

    def forward(self, x):
        """
        Args:
            x: [B, T, input_size]
        Returns:
            out:        [B, T, input_size]
            attn_weights: [B, T, T]
            log_probs:  [B, T]
            value:      [B, 1]
        """
        out, _ = self.lstm(x)                           # [B, T, hidden_dim]

        # Atenção Q/K/V escalonada
        q = self.q_proj(out)
        k = self.k_proj(out)
        v = self.v_proj(out)

        scale        = out.size(-1) ** 0.5
        attn_scores  = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, T, T]
        attn_weights = torch.softmax(attn_scores, dim=-1)

        # Modulação por RL: ações binárias mascaram frames menos relevantes
        actions, log_probs, value = self.rl_attn(out, attn_weights)

        # Repondera os pesos de atenção com a máscara aprendida pelo actor
        attn_weights = attn_weights * (actions.unsqueeze(1) + 1e-6)
        attn_weights = torch.softmax(attn_weights, dim=-1)

        attn_output = torch.matmul(attn_weights, v)
        out = out + attn_output

        # Gate adaptativo SAOA
        attn_output = self.sig(self.saoa1(out)) * self.saoa2(out)
        out = out + attn_output

        # Gates de balanço de memória (competição normalizada entre i, f, o)
        i_gate = torch.exp(self.i_gate(out))
        f_gate = torch.exp(self.f_gate(out))
        o_gate = torch.exp(self.o_gate(out))

        gate_sum = i_gate + f_gate + o_gate + 1e-6
        i_gate   = i_gate / gate_sum
        f_gate   = f_gate / gate_sum
        o_gate   = o_gate / gate_sum

        c_gate = torch.tanh(self.c_gate(out))

        out = f_gate * out + i_gate * c_gate
        out = o_gate * out

        out = self.layer_norm(out)
        out = self.drop(out)
        out = self.fc(out)

        return out, attn_weights, log_probs, value


if __name__ == '__main__':
    pass