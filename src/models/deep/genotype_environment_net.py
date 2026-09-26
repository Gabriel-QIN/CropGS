"""A regularised marker-block neural model for genomic selection.

There is no chromosome/position metadata in the released files, so the
encoder uses deterministic contiguous marker blocks rather than pretending
that adjacent CSV rows are physically adjacent loci.  The genotype missing
mask is a separate input channel.  Weather features condition the genotype
embedding through a learned gate, which gives the model a compact G x E
interaction without a full marker-by-weather tensor.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class GenotypeEnvironmentNet(nn.Module):
    """DeepGxE-Net-L: block encoder + weather-conditioned interaction head."""

    def __init__(
        self,
        n_markers: int,
        n_environment_features: int,
        block_size: int = 256,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 3,
        dropout: float = 0.20,
        head_hidden: int = 256,
    ) -> None:
        super().__init__()
        if n_markers <= 0 or n_environment_features <= 0:
            raise ValueError("Both genotype and environment inputs must be non-empty")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_markers = int(n_markers)
        self.block_size = int(block_size)
        self.n_blocks = int(math.ceil(n_markers / block_size))
        padded_markers = self.n_blocks * block_size
        self.padded_markers = padded_markers

        self.marker_encoder = nn.Conv1d(2, d_model, kernel_size=block_size, stride=block_size, bias=False)
        self.marker_norm = nn.LayerNorm(d_model)
        self.position = nn.Parameter(torch.zeros(1, self.n_blocks, d_model))
        nn.init.normal_(self.position, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # The model has no padding mask at block level (all blocks are real
        # marker blocks, with only the final block zero-padded), so nested
        # tensor conversion is unnecessary and produces a noisy warning on
        # pre-norm encoder layers.
        self.marker_transformer = nn.TransformerEncoder(layer, num_layers=n_layers, enable_nested_tensor=False)
        self.marker_attention = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        env_hidden = max(128, d_model // 2)
        self.environment_encoder = nn.Sequential(
            nn.Linear(n_environment_features, env_hidden),
            nn.LayerNorm(env_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(env_hidden, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )
        self.environment_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(3 * d_model, head_hidden),
            nn.LayerNorm(head_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, dosage: torch.Tensor, missing_mask: torch.Tensor, environment: torch.Tensor) -> torch.Tensor:
        if dosage.ndim != 2 or missing_mask.ndim != 2:
            raise ValueError("dosage and missing_mask must be [batch, markers]")
        if dosage.shape != missing_mask.shape:
            raise ValueError("dosage and missing_mask shapes must match")
        if dosage.shape[1] != self.n_markers:
            raise ValueError(f"expected {self.n_markers} markers, got {dosage.shape[1]}")
        if self.padded_markers > self.n_markers:
            pad = self.padded_markers - self.n_markers
            dosage = torch.nn.functional.pad(dosage, (0, pad))
            missing_mask = torch.nn.functional.pad(missing_mask, (0, pad), value=1.0)
        marker_input = torch.stack((dosage, missing_mask), dim=1)
        tokens = self.marker_encoder(marker_input).transpose(1, 2)
        tokens = self.marker_norm(tokens) + self.position
        tokens = self.marker_transformer(tokens)
        attention = torch.softmax(self.marker_attention(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        genotype_embedding = torch.sum(tokens * attention, dim=1)

        environment_embedding = self.environment_encoder(environment)
        interaction = genotype_embedding * self.environment_gate(environment_embedding)
        fused = torch.cat((genotype_embedding, environment_embedding, interaction), dim=1)
        return self.fusion(fused).squeeze(-1)

    def parameter_count(self) -> int:
        return int(sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad))


class ReactionNormNet(GenotypeEnvironmentNet):
    """Reaction-norm decomposition with explicit G, E, and G x E heads.

    The additive genotype path is not forced through the environment gate.
    This gives the model a stable genomic fallback when a new environment is
    far outside the weather support, while a low-rank interaction head can
    still learn environment-specific reranking.
    """

    def __init__(
        self,
        n_markers: int,
        n_environment_features: int,
        block_size: int = 256,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 3,
        dropout: float = 0.20,
        head_hidden: int = 256,
        interaction_rank: int = 64,
    ) -> None:
        super().__init__(n_markers, n_environment_features, block_size, d_model, n_heads, n_layers, dropout, head_hidden)
        # Replace the entangled fusion head with an identifiable reaction-norm
        # decomposition. The interaction starts small to reduce early fitting
        # of environment-specific noise.
        del self.fusion
        self.genotype_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.environment_head = nn.Sequential(nn.LayerNorm(d_model), nn.Dropout(dropout), nn.Linear(d_model, 1))
        self.genotype_interaction = nn.Linear(d_model, interaction_rank, bias=False)
        self.environment_interaction = nn.Linear(d_model, interaction_rank, bias=False)
        self.interaction_head = nn.Sequential(
            nn.LayerNorm(interaction_rank),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(interaction_rank, max(16, interaction_rank // 2)),
            nn.GELU(),
            nn.Linear(max(16, interaction_rank // 2), 1),
        )
        self.interaction_log_scale = nn.Parameter(torch.tensor(-2.0))

    def encode(self, dosage: torch.Tensor, missing_mask: torch.Tensor, environment: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.padded_markers > self.n_markers:
            pad = self.padded_markers - self.n_markers
            dosage = torch.nn.functional.pad(dosage, (0, pad))
            missing_mask = torch.nn.functional.pad(missing_mask, (0, pad), value=1.0)
        marker_input = torch.stack((dosage, missing_mask), dim=1)
        tokens = self.marker_encoder(marker_input).transpose(1, 2)
        tokens = self.marker_norm(tokens) + self.position
        tokens = self.marker_transformer(tokens)
        attention = torch.softmax(self.marker_attention(tokens).squeeze(-1), dim=1).unsqueeze(-1)
        genotype_embedding = torch.sum(tokens * attention, dim=1)
        return genotype_embedding, self.environment_encoder(environment)

    def forward(self, dosage: torch.Tensor, missing_mask: torch.Tensor, environment: torch.Tensor) -> torch.Tensor:
        if dosage.ndim != 2 or dosage.shape != missing_mask.shape or dosage.shape[1] != self.n_markers:
            raise ValueError("invalid genotype tensor shape")
        genotype_embedding, environment_embedding = self.encode(dosage, missing_mask, environment)
        g_main = self.genotype_head(genotype_embedding).squeeze(-1)
        e_main = self.environment_head(environment_embedding).squeeze(-1)
        interaction = self.genotype_interaction(genotype_embedding) * self.environment_interaction(environment_embedding)
        interaction = self.interaction_head(interaction).squeeze(-1)
        return g_main + e_main + torch.exp(self.interaction_log_scale) * interaction
