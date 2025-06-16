# models/patchtst.py
import torch
import torch.nn as nn

# PatchTST: transformer model for time series forecasting
class PatchTST(nn.Module):
    def __init__(
        self,
        in_channels: int,
        seq_len: int = 60,
        patch_size: int = 12,
        emb_dim: int = 64,
        n_heads: int = 4,
        ff_mult: int = 4,
        n_layers: int = 3,
        dropout: float = 0.1,
        channel_independent: bool = True,
        out_horizon: int = 1,
    ):
        super().__init__()
        # sequence length must be divisible by patch size
        assert seq_len % patch_size == 0, "seq_len must be a multiple of patch_size"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.channel_independent = channel_independent

        # patch embedding; project each patch to embedding space
        if channel_independent:
            self.patch_proj = nn.ModuleList([
                nn.Conv1d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
                for _ in range(in_channels)
            ])
        else:
            self.patch_proj = nn.Conv1d(
                in_channels, emb_dim, kernel_size=patch_size, stride=patch_size
            )

        # positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # prediction head; normalization and output layers
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, T = x.shape

        # patchify & embed; convert input to patch embeddings
        if self.channel_independent:
            patches = []
            for c in range(C):
                p = self.patch_proj[c](x[:, c:c+1, :])
                patches.append(p.permute(0, 2, 1))
            u = torch.stack(patches, dim=1).flatten(0, 1)
        else:
            p = self.patch_proj(x)
            u = p.permute(0, 2, 1)

        # more positional embeddings
        u = u + self.pos_embed

        # transformer encoding
        u = self.encoder(u)

        # pools across patches; apply prediction head
        z = u.mean(dim=1)
        z = self.norm(z)
        out = self.head(z)

        # reshape output if channel_independent
        if self.channel_independent:
            out = out.view(B, C, -1)
        return out.squeeze(-1) if C == 1 else out
