# PatchTST Multivariate Model
"""
PatchTST for multivariate time series forecasting.
This model is designed for multivariate input (in_channels > 1).
"""
import torch
import torch.nn as nn

# PatchTST: transformer model for multivariate time series forecasting
class PatchTST(nn.Module):
    """
    Multivariate PatchTST: Transformer model for multivariate time series forecasting.
    Output shape:
      - channel_independent=True: (batch, channels, out_horizon)
      - channel_independent=False: (batch, out_horizon)
    """
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
        if in_channels < 2:
            raise ValueError("patchtst_multivar.py is intended for multivariate input (in_channels > 1)")
        assert seq_len % patch_size == 0, "seq_len must be a multiple of patch_size"
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.channel_independent = channel_independent

        # Patch embedding: project each patch to embedding space
        if channel_independent:
            # Each channel gets its own Conv1d
            self.patch_proj = nn.ModuleList([
                nn.Conv1d(1, emb_dim, kernel_size=patch_size, stride=patch_size)
                for _ in range(in_channels)
            ])
        else:
            # All channels processed together
            self.patch_proj = nn.Conv1d(
                in_channels, emb_dim, kernel_size=patch_size, stride=patch_size
            )

        # Positional embeddings for each patch
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, emb_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            nhead=n_heads,
            dim_feedforward=emb_dim * ff_mult,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Prediction head: normalization and output layers
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Sequential(
            nn.Linear(emb_dim, emb_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, out_horizon)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels, seq_len)
        Returns:
          - (batch, channels, out_horizon) if channel_independent
          - (batch, out_horizon) if not channel_independent
        """
        B, C, T = x.shape
        # Patchify & embed
        if self.channel_independent:
            # Each channel processed separately
            patches = []
            for c in range(C):
                p = self.patch_proj[c](x[:, c:c+1, :])
                patches.append(p.permute(0, 2, 1))
            u = torch.stack(patches, dim=1).flatten(0, 1)  # (B*C, n_patches, emb_dim)
        else:
            # All channels processed together
            p = self.patch_proj(x)
            u = p.permute(0, 2, 1)  # (B, n_patches, emb_dim)

        # Add positional embeddings
        u = u + self.pos_embed
        u = self.encoder(u)
        z = u.mean(dim=1)
        z = self.norm(z)
        out = self.head(z)

        # Reshape output for channel_independent mode
        if self.channel_independent:
            out = out.view(B, C, -1)
            return out  # (B, C, out_horizon)
        else:
            return out  # (B, out_horizon)
