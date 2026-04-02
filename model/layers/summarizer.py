import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import sLSTM, mLSTM, SEBlock


class xLSTM(nn.Module):
    """Orquestrador dos blocos sLSTM e mLSTM com fusão gated e recalibração SE.

    Arquitetura:
      [input]
         │
       sLSTM  ──── extração local + recorrência escalar com gates de memória
         │
       mLSTM  ──── recorrência matricial + atenção Q/K/V + modulação RL
         │
       Gating ──── combinação aprendível: gate * sLSTM + (1-gate) * mLSTM
         │
      SEBlock ──── recalibração de canais (atenção sobre features, não tempo)
         │
      fc + fc_output ── projeção para [T, 1] de scores de importância
         │
      sigmoid ──── normaliza saída para [0,1] (compatível com gtscore do dataset)

    Nota sobre nomenclatura: sLSTM e mLSTM são implementações originais
    inspiradas conceitualmente no xLSTM, mas com estrutura de gates e
    mecanismo de balanço de memória próprios. Não seguem o paper xLSTM.
    """

    def __init__(self, input_size, output_size, num_segments,
                 hidden_dim=512, num_layers=2, dropout=0.5,
                 max_seq_len=200, pos_enc='absolute'):
        super(xLSTM, self).__init__()

        self.num_segments = num_segments
        self.pos_enc_type = pos_enc

        self.slstm = sLSTM(input_size, hidden_dim, dropout=dropout)
        self.mlstm = mLSTM(
            input_size,
            hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_seq_len=max_seq_len,          # FIX — propagado do config
        )
        self.se_block = SEBlock(input_size)

        # FIX — self.conv estava definido mas nunca usado no forward().
        # Removido para evitar parâmetros mortos que consomem memória e
        # aparecem no state_dict sem contribuir para o modelo.

        # Gate de fusão entre sLSTM e mLSTM (aprendível por posição)
        self.attn_linear   = nn.Linear(input_size, input_size)
        self.attn_softmax  = nn.Softmax(dim=-1)   # mantido para uso opcional

        self.fc        = nn.Linear(input_size, output_size)
        self.fc_output = nn.Linear(output_size, 1)

        # FIX — Positional encoding: estava configurado em configs.py com
        # default 'absolute' mas nunca aplicado no forward(). Implementado
        # como Embedding aprendível sobre posições 0..max_pos-1.
        # Desativado quando pos_enc == 'none'.
        self.pos_embedding = None
        if self.pos_enc_type != 'none':
            # 5000 posições cobre vídeos de até ~41 min a 2 fps
            self.pos_embedding = nn.Embedding(5000, input_size)

    def forward(self, x):
        """
        Args:
            x: [T, F] — features de um vídeo (sem dimensão de batch)
               ou [1, T, F]

        Returns:
            output:       [1, T]  — scores de importância por frame (sigmoid)
            attn_weights: [1, T, T]
            log_probs:    [1, T]
            value:        [1, 1]
        """
        # Garante shape [1, T, F] para os módulos internos
        if x.dim() == 2:
            x = x.unsqueeze(0)                       # [1, T, F]

        # FIX — aplicação do positional encoding (antes ausente)
        # Injeta informação de posição temporal nas features antes dos LSTMs.
        # Impacto: ajuda o modelo a distinguir início/meio/fim do vídeo,
        # relevante pois a importância de frames pode depender da posição.
        if self.pos_embedding is not None:
            T = x.size(1)
            positions = torch.arange(T, device=x.device)     # [T]
            pos_enc   = self.pos_embedding(positions)         # [T, F]
            x = x + pos_enc.unsqueeze(0)                     # [1, T, F]

        x_slstm = self.slstm(x)                              # [1, T, F]
        x_mlstm, attn_weights, log_probs, value = self.mlstm(x_slstm)

        # Gating aprendível: quanto de sLSTM vs mLSTM usar por posição
        gate      = torch.sigmoid(self.attn_linear(x_slstm + x_mlstm))
        x_combined = gate * x_slstm + (1 - gate) * x_mlstm  # [1, T, F]

        # SEBlock opera em [B, C, T] → permuta antes e depois
        x_se = self.se_block(x_combined.permute(0, 2, 1)).permute(0, 2, 1)

        output = self.fc(x_se)                               # [1, T, output_size]
        output = self.fc_output(output)                      # [1, T, 1]

        # FIX — sigmoid normaliza os scores para [0, 1], compatível com
        # os gtscore do dataset (também normalizados entre 0 e 1).
        # Antes: saída sem bounds dificultava convergência do MSELoss.
        output = torch.sigmoid(output)

        output = output.view(output.size(0), -1)             # [1, T]

        return output, attn_weights, log_probs, value


if __name__ == '__main__':
    pass