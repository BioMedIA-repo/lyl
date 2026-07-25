from typing import Dict, Optional, Union
import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import softmax
import pytorch_lightning as pl

class DirectGeneHead(nn.Module):
    """Original V0 direct prediction head."""

    def __init__(self, hidden_dim: int, num_genes: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_genes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class SpatialDiffusionBank(nn.Module):
    """Build multi-scale row-normalized spatial diffusion embeddings."""

    def __init__(self, num_diffusion_steps: int = 3, diffusion_residual: float = 0.0):
        super().__init__()
        if num_diffusion_steps < 0:
            raise ValueError("num_diffusion_steps must be non-negative.")
        if not 0.0 <= diffusion_residual <= 1.0:
            raise ValueError("diffusion_residual must be in [0, 1].")
        self.num_diffusion_steps = int(num_diffusion_steps)
        self.diffusion_residual = float(diffusion_residual)

    @staticmethod
    def diffuse_once(h: torch.Tensor, edge_index_low: torch.Tensor, edge_weight_low: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index_low
        w = edge_weight_low.to(device=h.device, dtype=h.dtype)
        out = h.new_zeros(h.size(0), h.size(1))
        deg = h.new_zeros(h.size(0))
        out.index_add_(0, dst, h[src] * w.unsqueeze(-1))
        deg.index_add_(0, dst, w)
        return out / deg.clamp_min(1e-6).unsqueeze(-1)

    def forward(self, h: torch.Tensor, edge_index_low: torch.Tensor, edge_weight_low: torch.Tensor) -> torch.Tensor:
        h_list = [h]
        h_prev = h
        for _ in range(self.num_diffusion_steps):
            ph = self.diffuse_once(h_prev, edge_index_low, edge_weight_low)
            if self.diffusion_residual > 0:
                h_next = (1.0 - self.diffusion_residual) * ph + self.diffusion_residual * h_prev
            else:
                h_next = ph
            h_list.append(h_next)
            h_prev = h_next
        return torch.stack(h_list, dim=0)


class GeneSpecificScaleAttention(nn.Module):
    """Gene-specific attention over diffusion scales."""

    def __init__(
        self,
        num_genes: int,
        num_scales: int,
        gene_embed_dim: int = 64,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive.")
        self.num_genes = num_genes
        self.num_scales = num_scales
        self.temperature = float(temperature)

        self.gene_embedding = nn.Embedding(num_genes, gene_embed_dim)
        self.scale_mlp = nn.Sequential(
            nn.Linear(gene_embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_scales),
        )

    def forward(self, device: Optional[torch.device] = None) -> torch.Tensor:
        if device is None:
            device = self.gene_embedding.weight.device
        gene_ids = torch.arange(self.num_genes, device=device)
        e = self.gene_embedding(gene_ids)
        logits = self.scale_mlp(e) / self.temperature
        return F.softmax(logits, dim=-1)

    def get_expected_scale(self, pi: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pi is None:
            pi = self.forward()
        scales = torch.arange(self.num_scales, device=pi.device, dtype=pi.dtype)
        return (pi * scales.unsqueeze(0)).sum(dim=-1)


class GeneSpecificDiffusionHead(nn.Module):
    """Predict gene-wise diffusion residuals from a multi-scale embedding bank."""

    def __init__(
        self,
        hidden_dim: int,
        num_genes: int,
        pred_dim: Optional[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        pred_dim = hidden_dim if pred_dim is None else pred_dim
        self.pred_dim = pred_dim
        self.scale_proj = nn.Sequential(
            nn.Linear(hidden_dim, pred_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gene_readout = nn.Parameter(torch.empty(num_genes, pred_dim))
        self.gene_bias = nn.Parameter(torch.zeros(num_genes))
        nn.init.normal_(self.gene_readout, mean=0.0, std=0.02)

    def forward(self, h_stack: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
        z_stack = self.scale_proj(h_stack)
        z_gene = torch.einsum("gs,snd->ngd", pi, z_stack)
        return torch.einsum("ngd,gd->ng", z_gene, self.gene_readout) + self.gene_bias


# ============================================================================
# V0 Encoder: SpotGNNEncoder (GATv2/GCN)
# ============================================================================


class SpotGNNEncoder(nn.Module):
    """V0-style spot GNN encoder: input projection + GATv2/GCN layers."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        conv_type: str = "gatv2",
        heads: int = 4,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.conv_type = conv_type

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            if conv_type == "gatv2":
                from torch_geometric.nn import GATv2Conv

                conv = GATv2Conv(
                    hidden_dim,
                    hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                )
            elif conv_type == "gcn":
                from torch_geometric.nn import GCNConv

                conv = GCNConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unsupported conv_type: {conv_type}. Use 'gatv2' or 'gcn'.")
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, edge_index)
            h_next = norm(h_next)
            h_next = self.act(h_next)
            h_next = self.drop(h_next)
            if h_next.shape == h.shape:
                h = h + h_next
            else:
                h = h_next
        return h



class SFAGNNLayer(nn.Module):
    """Multi-scale spatial frequency-adaptive filter bank layer."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.1,
        use_position: bool = True,
        pos_rbf_dim: int = 8,
        pos_mlp_hidden: int = 32,
        aggregation: str = "attention",
        gate_hidden_dim: Optional[int] = None,
        gate_type: str = "scalar",
        sfa_mode: str = "full",
        eps: float = 1e-8,
    ):
        super().__init__()
        if aggregation not in {"attention", "mean"}:
            raise ValueError("aggregation must be 'attention' or 'mean'.")
        if gate_type not in {"scalar", "feature"}:
            raise ValueError("gate_type must be 'scalar' or 'feature'.")
        if sfa_mode not in {"low_only", "high_only", "old_sfa", "multi_low", "multi_high", "full"}:
            raise ValueError("Unsupported sfa_mode.")

        self.hidden_dim = hidden_dim
        self.use_position = bool(use_position)
        self.pos_rbf_dim = int(pos_rbf_dim)
        self.pos_mlp_hidden = int(pos_mlp_hidden)
        self.aggregation = aggregation
        self.gate_type = gate_type
        self.sfa_mode = sfa_mode
        self.eps = float(eps)
        self.rbf_gamma = 10.0
        self.component_names = self._component_names_for_mode(sfa_mode)
        num_components = len(self.component_names)

        self.component_projs = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_components)]
        )

        if self.use_position:
            self.pos_mlp = nn.Sequential(
                nn.Linear(pos_rbf_dim, pos_mlp_hidden),
                nn.GELU(),
                nn.Linear(pos_mlp_hidden, pos_mlp_hidden),
                nn.GELU(),
            )
            att_pos_in_dim = 2 * hidden_dim + pos_mlp_hidden
        else:
            self.pos_mlp = None
            att_pos_in_dim = None

        self.att_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.att_pos_mlp = None
        if att_pos_in_dim is not None:
            self.att_pos_mlp = nn.Sequential(
                nn.Linear(att_pos_in_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        gate_hidden_dim = hidden_dim if gate_hidden_dim is None else gate_hidden_dim
        self.selector_mlp = nn.Sequential(
            nn.Linear(5 * hidden_dim, gate_hidden_dim),
            nn.GELU(),
            nn.Linear(gate_hidden_dim, num_components),
        )
        nn.init.zeros_(self.selector_mlp[-1].bias)

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gamma_raw = nn.Parameter(torch.tensor(-5.0))
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    @staticmethod
    def _component_names_for_mode(sfa_mode: str):
        if sfa_mode == "low_only":
            return ["L1"]
        if sfa_mode == "high_only":
            return ["B1"]
        if sfa_mode == "old_sfa":
            return ["L1", "B1"]
        if sfa_mode == "multi_low":
            return ["L1", "L2"]
        if sfa_mode == "multi_high":
            return ["B1", "B2"]
        if sfa_mode == "full":
            return ["L1", "L2", "B1", "B2"]
        raise ValueError(f"Unsupported sfa_mode: {sfa_mode}")

    def _position_encoding(
        self,
        pos: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
    ) -> torch.Tensor:
        delta = pos[dst] - pos[src]
        dist = torch.linalg.norm(delta, dim=-1)
        nonzero = dist > 0
        if nonzero.any():
            scale = dist[nonzero].mean().detach().clamp_min(self.eps)
        else:
            scale = dist.new_tensor(1.0)
        dist_norm = (dist / scale).clamp(max=3.0)
        centers = torch.linspace(
            0.0,
            3.0,
            self.pos_rbf_dim,
            device=pos.device,
            dtype=pos.dtype,
        )
        rbf = torch.exp(-self.rbf_gamma * (dist_norm.unsqueeze(-1) - centers.unsqueeze(0)).pow(2))
        return self.pos_mlp(rbf)

    @staticmethod
    def _scatter_add(messages: torch.Tensor, dst: torch.Tensor, num_nodes: int) -> torch.Tensor:
        out = messages.new_zeros((num_nodes, messages.size(-1)))
        out.index_add_(0, dst, messages)
        return out

    def _attention_weights(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        pos: Optional[torch.Tensor],
        num_nodes: int,
    ) -> torch.Tensor:
        h_src = h[src]
        h_dst = h[dst]
        if self.use_position and pos is not None and self.att_pos_mlp is not None:
            pos_enc = self._position_encoding(pos, src, dst)
            att_input = torch.cat([h_src, h_dst, pos_enc], dim=-1)
            edge_score = self.att_pos_mlp(att_input).squeeze(-1)
        else:
            # Without coordinates, attention depends only on source/destination features.
            att_input = torch.cat([h_src, h_dst], dim=-1)
            edge_score = self.att_mlp(att_input).squeeze(-1)
        return softmax(edge_score, dst, num_nodes=num_nodes)

    def _propagate(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return P_h, a normalized h_src -> dst neighborhood propagation."""
        src, dst, weights = self._build_propagation_operator(h, edge_index, edge_weight, pos)
        return self._apply_propagation(h, src, dst, weights)

    def _build_propagation_operator(
        self,
        h: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ):
        """Build P from the current layer state and graph."""
        num_nodes = h.size(0)
        src, dst = edge_index[0], edge_index[1]
        edge_weight = edge_weight.to(device=h.device, dtype=h.dtype).clamp_min(self.eps)
        if self.aggregation == "attention":
            att = self._attention_weights(h, src, dst, pos, num_nodes)
            weights = att * edge_weight
            deg = h.new_zeros((num_nodes,))
            deg.index_add_(0, dst, weights)
            weights = weights / deg[dst].clamp_min(self.eps)
        else:
            deg = h.new_zeros((num_nodes,))
            deg.index_add_(0, dst, edge_weight)
            weights = edge_weight / deg[dst].clamp_min(self.eps)
        return src, dst, weights

    def _apply_propagation(
        self,
        h: torch.Tensor,
        src: torch.Tensor,
        dst: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a previously built P to h."""
        messages = weights.unsqueeze(-1) * h[src]
        num_nodes = h.size(0)
        out = self._scatter_add(messages, dst, num_nodes)
        return out

    def forward(
        self,
        h: torch.Tensor,
        edge_index_low: torch.Tensor,
        edge_weight_low: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_weight_high: torch.Tensor,
        pos: Optional[torch.Tensor] = None,
    ):
        src, dst, weights = self._build_propagation_operator(h, edge_index_low, edge_weight_low, pos)
        L1 = self._apply_propagation(h, src, dst, weights)
        L2 = self._apply_propagation(L1, src, dst, weights)
        B1 = h - L1

        B2_neighbor = self._propagate(h, edge_index_high, edge_weight_high, pos)
        B2 = B2_neighbor - h

        component_map = {"L1": L1, "L2": L2, "B1": B1, "B2": B2}
        names = self.component_names
        components = [component_map[name] for name in names]
        selector_input = torch.cat([h, L1, L2, B1, B2], dim=-1)
        selector_logits = self.selector_mlp(selector_input)
        freq_weights = torch.softmax(selector_logits, dim=-1)

        mixed = h.new_zeros(h.shape)
        for m, component in enumerate(components):
            mixed = mixed + freq_weights[:, m:m + 1] * self.component_projs[m](component)

        gamma = F.softplus(self.gamma_raw)
        mixed = self.out_proj(mixed)
        h_new = self.norm(h + gamma * self.dropout(mixed))
        h_new = self.act(h_new)

        aux = {
            "gate_mean": freq_weights.mean().detach(),
            "gate_std": freq_weights.std(unbiased=False).detach(),
            "gamma": gamma.detach(),
            "freq_weight_mean": freq_weights.mean(dim=0).detach(),
            "freq_component_names": names,
        }
        return h_new, aux


class SFAGNNEncoder(nn.Module):
    """SFA-GNN encoder with the same output shape as SpotGNNEncoder."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_position: bool = True,
        pos_rbf_dim: int = 8,
        aggregation: str = "attention",
        gate_type: str = "scalar",
        sfa_mode: str = "full",
    ):
        super().__init__()
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                SFAGNNLayer(
                    hidden_dim=hidden_dim,
                    dropout=dropout,
                    use_position=use_position,
                    pos_rbf_dim=pos_rbf_dim,
                    aggregation=aggregation,
                    gate_type=gate_type,
                    sfa_mode=sfa_mode,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self,
        x: torch.Tensor,
        pos: Optional[torch.Tensor],
        edge_index_low: torch.Tensor,
        edge_weight_low: torch.Tensor,
        edge_index_high: torch.Tensor,
        edge_weight_high: torch.Tensor,
        return_aux: bool = False,
    ):
        h = self.input_proj(x)
        gate_means = []
        gate_stds = []
        gammas = []
        freq_weight_means = []
        freq_component_names = None
        for layer in self.layers:
            h, aux_l = layer(h, edge_index_low, edge_weight_low, edge_index_high, edge_weight_high, pos)
            gate_means.append(aux_l["gate_mean"])
            gate_stds.append(aux_l["gate_std"])
            gammas.append(aux_l["gamma"])
            freq_weight_means.append(aux_l["freq_weight_mean"])
            freq_component_names = aux_l["freq_component_names"]

        if not return_aux:
            return h
        return {
            "H": h,
            "gate_mean_per_layer": gate_means,
            "gate_std_per_layer": gate_stds,
            "gamma_per_layer": gammas,
            "freq_weight_mean_per_layer": freq_weight_means,
            "freq_component_names": freq_component_names,
        }




class SFAGSDiffGNN(nn.Module):
    """GSDiff-GNN with only the encoder replaced by SFAGNNEncoder."""

    def __init__(
        self,
        in_dim: int,
        num_genes: int,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        num_diffusion_steps: int = 3,
        diffusion_residual: float = 0.0,
        gene_embed_dim: int = 64,
        scale_attn_hidden_dim: int = 128,
        scale_temperature: float = 1.0,
        pred_dim: Optional[int] = None,
        alpha_mode: str = "softplus",
        alpha_init: float = -5.0,
        residual_alpha: float = 0.1,
        use_direct_branch: bool = True,
        diffusion_mode: str = "gene_specific",
        use_position: bool = True,
        pos_rbf_dim: int = 8,
        sfa_aggregation: str = "attention",
        gate_type: str = "scalar",
        sfa_mode: str = "full",
        conv_type: str = "gatv2",
        heads: int = 4,
    ):
        super().__init__()
        if diffusion_mode not in {"none", "shared", "gene_specific"}:
            raise ValueError("diffusion_mode must be one of: none, shared, gene_specific.")
        if sfa_mode not in {"no_sfa", "low_only", "high_only", "old_sfa", "multi_low", "multi_high", "full"}:
            raise ValueError("Unsupported sfa_mode.")
        if not use_direct_branch and diffusion_mode == "none":
            raise ValueError("At least one prediction branch must be enabled.")
        if alpha_mode not in {"softplus", "sigmoid", "fixed"}:
            raise ValueError("alpha_mode must be one of: softplus, sigmoid, fixed.")

        self.num_genes = num_genes
        self.num_diffusion_steps = num_diffusion_steps
        self.num_scales = num_diffusion_steps + 1
        self.alpha_mode = alpha_mode
        self.residual_alpha = float(residual_alpha)
        self.use_direct_branch = bool(use_direct_branch)
        self.diffusion_mode = diffusion_mode
        self.sfa_mode = sfa_mode

        if self.sfa_mode == "no_sfa":
            self.encoder = SpotGNNEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                conv_type=conv_type,
                heads=heads,
            )
        else:
            self.encoder = SFAGNNEncoder(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_position=use_position,
                pos_rbf_dim=pos_rbf_dim,
                aggregation=sfa_aggregation,
                gate_type=gate_type,
                sfa_mode=sfa_mode,
            )
        self.direct_head = DirectGeneHead(hidden_dim, num_genes, dropout)
        self.diffusion_bank = SpatialDiffusionBank(num_diffusion_steps, diffusion_residual)
        self.diffusion_head = GeneSpecificDiffusionHead(hidden_dim, num_genes, pred_dim, dropout)

        if self.diffusion_mode == "shared":
            self.pi_shared = nn.Parameter(torch.zeros(self.num_scales))
            self.scale_attention = None
        elif self.diffusion_mode == "gene_specific":
            self.pi_shared = None
            self.scale_attention = GeneSpecificScaleAttention(
                num_genes=num_genes,
                num_scales=self.num_scales,
                gene_embed_dim=gene_embed_dim,
                hidden_dim=scale_attn_hidden_dim,
                temperature=scale_temperature,
            )
        else:
            self.pi_shared = None
            self.scale_attention = None

        if alpha_mode == "fixed":
            self.register_parameter("alpha_raw", None)
        else:
            self.alpha_raw = nn.Parameter(torch.tensor(float(alpha_init)))

    def get_alpha(self) -> torch.Tensor:
        if self.alpha_mode == "softplus":
            return F.softplus(self.alpha_raw)
        if self.alpha_mode == "sigmoid":
            return torch.sigmoid(self.alpha_raw)
        return torch.tensor(self.residual_alpha, device=next(self.parameters()).device)

    def get_scale_attention(self, device: torch.device) -> torch.Tensor:
        if self.diffusion_mode == "none":
            return None
        if self.diffusion_mode == "shared":
            pi = F.softmax(self.pi_shared, dim=0).unsqueeze(0)
            return pi.expand(self.num_genes, self.num_scales)
        return self.scale_attention(device=device)

    def get_expected_scale(self, pi: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if pi is None:
            return None
        scales = torch.arange(self.num_scales, device=pi.device, dtype=pi.dtype)
        return (pi * scales.unsqueeze(0)).sum(dim=-1)

    def forward(
        self,
        data,
        return_aux: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        edge_index_low = data.edge_index_low
        edge_weight_low = data.edge_weight_low
        edge_index_high = data.edge_index_high
        edge_weight_high = data.edge_weight_high

        if self.sfa_mode == "no_sfa":
            h = self.encoder(data.x, edge_index_low)
            encoder_outputs = {
                "gate_mean_per_layer": [],
                "gate_std_per_layer": [],
                "gamma_per_layer": [],
                "freq_weight_mean_per_layer": [],
                "freq_component_names": [],
            }
        else:
            encoder_outputs = self.encoder(
                x=data.x,
                pos=data.pos,
                edge_index_low=edge_index_low,
                edge_weight_low=edge_weight_low,
                edge_index_high=edge_index_high,
                edge_weight_high=edge_weight_high,
                return_aux=True,
            )
            h = encoder_outputs["H"]
        y_base = self.direct_head(h) if self.use_direct_branch else h.new_zeros((h.size(0), self.num_genes))

        alpha = self.get_alpha()
        if self.diffusion_mode == "none":
            h_stack = None
            pi = None
            expected_scale = None
            y_diff = torch.zeros_like(y_base)
            y_pred = y_base
        else:
            h_stack = self.diffusion_bank(h, edge_index_low, edge_weight_low)
            pi = self.get_scale_attention(h.device)
            expected_scale = self.get_expected_scale(pi)
            y_diff = self.diffusion_head(h_stack, pi)
            y_pred = y_base + alpha * y_diff

        if not return_aux:
            return y_pred

        return {
            "y_pred": y_pred,
            "y_base": y_base,
            "y_diff": y_diff,
            "H": h,
            "H_stack": h_stack,
            "pi": pi,
            "expected_scale": expected_scale,
            "alpha": alpha,
            "sfa_gate_mean_per_layer": encoder_outputs["gate_mean_per_layer"],
            "sfa_gate_std_per_layer": encoder_outputs["gate_std_per_layer"],
            "sfa_gamma_per_layer": encoder_outputs.get("gamma_per_layer"),
            "sfa_freq_weight_mean_per_layer": encoder_outputs.get("freq_weight_mean_per_layer"),
            "sfa_freq_component_names": encoder_outputs.get("freq_component_names"),
        }



# ============================================================================
# Roughness-guided target diffusion scale
# ============================================================================

def roughness_to_target_scale(
    roughness: torch.Tensor,
    num_diffusion_steps: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    if roughness.ndim != 1:
        raise ValueError(
            f"roughness must have shape [G], got {tuple(roughness.shape)}"
        )

    if num_diffusion_steps < 1:
        raise ValueError("num_diffusion_steps must be at least 1.")

    roughness = roughness.detach().float()

    # Replace NaN/Inf with the median finite roughness.
    finite_mask = torch.isfinite(roughness)
    if not finite_mask.any():
        return torch.full_like(
            roughness,
            fill_value=float(num_diffusion_steps) / 2.0,
        )

    finite_median = roughness[finite_mask].median()
    roughness = torch.where(finite_mask, roughness, finite_median)

    # Roughness should be non-negative.
    roughness = roughness.clamp_min(eps)

    # Log transform reduces heavy-tailed roughness differences.
    log_roughness = torch.log(roughness)

    # Robustly clip extreme genes.
    lower = torch.quantile(log_roughness, 0.05)
    upper = torch.quantile(log_roughness, 0.95)

    if (upper - lower).abs() < eps:
        return torch.full_like(
            roughness,
            fill_value=float(num_diffusion_steps) / 2.0,
        )

    log_roughness = log_roughness.clamp(min=lower, max=upper)

    # Normalize to [0, 1].
    roughness_norm = (
        log_roughness - lower
    ) / (
        upper - lower + eps
    )

    # Reverse mapping:
    # smoother gene (low roughness) -> larger diffusion scale
    # sharper gene (high roughness) -> smaller diffusion scale
    target_scale = (
        1.0 - roughness_norm
    ) * float(num_diffusion_steps)

    return target_scale.clamp(
        min=0.0,
        max=float(num_diffusion_steps),
    )

# ============================================================================
# Utility
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
