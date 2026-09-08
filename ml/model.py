"""ImportanceScorer — per-frame highlight scores from a CLIP feature sequence.

A one-layer bidirectional GRU over the frame features, then a small MLP head
to a scalar per frame, sigmoid to [0, 1]. This is the vsLSTM / dppLSTM family
without the determinantal-point-process layer: for ~75 training videos a
supervised regressor with heavy dropout generalises better than anything
larger, and it runs on CPU in milliseconds.

The checkpoint stores ``state_dict`` plus the hyperparameters, so
``ml.infer`` can rebuild the exact architecture with no config file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn


@dataclass
class ModelConfig:
    input_dim: int = 512          # CLIP ViT-B/32 image embedding width
    hidden_dim: int = 128
    num_layers: int = 1
    dropout: float = 0.5
    head_dim: int = 64


class ImportanceScorer(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        c = self.cfg
        self.input_norm = nn.LayerNorm(c.input_dim)
        self.gru = nn.GRU(
            c.input_dim, c.hidden_dim, num_layers=c.num_layers,
            batch_first=True, bidirectional=True,
            dropout=c.dropout if c.num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * c.hidden_dim, c.head_dim),
            nn.ReLU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.head_dim, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        """x: (B, T, input_dim) -> scores: (B, T) in [0, 1]."""
        h = self.input_norm(x)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                h, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            out, _ = self.gru(packed)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=x.size(1))
        else:
            out, _ = self.gru(h)
        return torch.sigmoid(self.head(out).squeeze(-1))

    # --- checkpoint helpers -------------------------------------------------

    def save(self, path) -> None:
        torch.save({"config": asdict(self.cfg), "state_dict": self.state_dict()}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "ImportanceScorer":
        blob = torch.load(path, map_location=map_location)
        model = cls(ModelConfig(**blob["config"]))
        model.load_state_dict(blob["state_dict"])
        model.eval()
        return model


def masked_ranking_loss(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, margin: float = 0.1,
) -> torch.Tensor:
    """Pairwise hinge: within each clip, sampled frame pairs should keep the
    human ordering. Complements MSE — MSE fixes the scale, this fixes the rank
    that the F1 / Kendall metrics actually reward.
    """
    B, T = pred.shape
    losses = []
    for b in range(B):
        m = mask[b].bool()
        p, t = pred[b][m], target[b][m]
        if p.numel() < 2:
            continue
        n = min(p.numel(), 64)
        idx = torch.randperm(p.numel(), device=p.device)[:n]
        pi, ti = p[idx], t[idx]
        dp = pi[:, None] - pi[None, :]
        dt = ti[:, None] - ti[None, :]
        sign = torch.sign(dt)
        pair = torch.clamp(margin - sign * dp, min=0.0)
        pair = pair[dt.abs() > 1e-6]
        if pair.numel():
            losses.append(pair.mean())
    if not losses:
        return pred.sum() * 0.0
    return torch.stack(losses).mean()
