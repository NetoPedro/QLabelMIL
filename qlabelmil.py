import torch
import torch.nn as nn

from typing import Optional, Dict, Tuple
from src.models.utils import BaseModel, SingleLayer_Classifier
import logging


class QueryDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        ffn_mult: int = 4,
        attn_dropout: float = 0.1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm_q1 = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )

        self.norm_q2 = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=attn_dropout, batch_first=True
        )

        self.norm_q3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(
        self,
        Q: torch.Tensor,
        H: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ):
        # Q: (B, C, d_model)   H: (B, N, d_model)
        q = self.norm_q1(Q)
        Q = Q + self.drop(self.self_attn(q, q, q, need_weights=False)[0])

        q = self.norm_q2(Q)
        ctx, attn = self.cross_attn(
            q,
            H,
            H,
            key_padding_mask=key_padding_mask,
            need_weights=True,
            average_attn_weights=True,
        )
        Q = Q + self.drop(ctx)

        Q = Q + self.drop(self.ffn(self.norm_q3(Q)))
        return Q, attn  # attn: (B, C, N)


class QueryLabelMIL(nn.Module):
    def __init__(
        self,
        in_dim: int,
        d_model: int = 512,
        n_labels: int = 6,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_mult: int = 4,
        attn_dropout: float = 0.1,
        class_dropout: float = 0.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = SingleLayer_Classifier(in_dim, d_model, dropout=class_dropout)
        self.in_norm = nn.LayerNorm(d_model)

        self.label_queries = nn.Parameter(torch.empty(n_labels, d_model))
        nn.init.normal_(self.label_queries, std=0.02)

        self.layers = nn.ModuleList(
            QueryDecoderLayer(d_model, n_heads, ffn_mult, attn_dropout, dropout)
            for _ in range(n_layers)
        )
        self.out_norm = nn.LayerNorm(d_model)

        self.cls_weight = nn.Parameter(torch.empty(n_labels, d_model))
        nn.init.normal_(self.cls_weight, std=0.02)
        self.cls_bias = nn.Parameter(torch.zeros(n_labels))

    def encode(self, H: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        # H: (B, N, in_dim)
        H = self.in_norm(self.in_proj(H))
        Q = self.label_queries.unsqueeze(0).expand(H.size(0), -1, -1)
        attn = None

        for layer in self.layers:
            Q, attn = layer(Q, H, key_padding_mask)

        Z = self.out_norm(Q)  # (B, C, d_model)
        return Z, attn  # attn: (B, C, N)

    def forward(self, H: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        Z, attn = self.encode(H, key_padding_mask)
        logits = (Z * self.cls_weight).sum(-1) + self.cls_bias  # (B, C)
        return logits, Z, attn


class GraphConvolution(nn.Module):
    """out = adj @ (x @ W) + b.  adj is a fixed (C, C) propagation matrix."""

    def __init__(self, in_dim: int, out_dim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        out = adj @ (x @ self.weight)  # (C, out_dim)
        if self.bias is not None:
            out = out + self.bias
        return out


class LabelGCN(nn.Module):
    def __init__(
        self,
        adjacency: torch.Tensor,
        node_dim: int = 300,
        hidden_dim: int = 512,
        out_dim: int = 512,
        node_init: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        C = adjacency.size(0)
        self.register_buffer("adj", adjacency.float())

        if node_init is None:
            self.nodes = nn.Parameter(torch.empty(C, node_dim))
            nn.init.normal_(self.nodes, std=0.02)
        else:  # e.g. text embeddings of label names
            self.nodes = nn.Parameter(node_init.clone().float())
            node_dim = node_init.size(1)

        self.gc1 = GraphConvolution(node_dim, hidden_dim)
        self.gc2 = GraphConvolution(hidden_dim, out_dim)
        self.act = nn.LeakyReLU(0.2)

    def forward(self) -> torch.Tensor:
        g = self.act(self.gc1(self.nodes, self.adj))
        g = self.gc2(g, self.adj)  # (C, out_dim)
        return g

    @staticmethod
    def build_adjacency(
        Y: torch.Tensor, tau: float = 0.4, p: float = 0.25
    ) -> torch.Tensor:
        Y = Y.float()
        co = Y.t() @ Y
        cnt = co.diag().clamp(min=1.0)
        cond = co / cnt.unsqueeze(0)  # cond[i, j] = P(label_i | label_j)

        A = (cond >= tau).float()
        A.fill_diagonal_(0.0)
        deg = A.sum(1, keepdim=True).clamp(min=1.0)
        A = A * (p / deg)  # neighbours of row i share weight p
        A.fill_diagonal_(1.0 - p)  # node keeps weight 1 - p on itself
        return A


class LabelCRFHead(nn.Module):
    def __init__(
        self,
        n_labels: int,
        n_iters: int = 2,
        init_potential: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        if init_potential is None:
            self.phi = nn.Parameter(torch.zeros(n_labels, n_labels))
        else:
            self.phi = nn.Parameter(init_potential.clone().float())
        self.n_iters = n_iters

    def forward(self, base_logits: torch.Tensor) -> torch.Tensor:
        phi = self.phi - torch.diag_embed(torch.diagonal(self.phi))
        s = base_logits
        p = torch.sigmoid(s)
        for _ in range(self.n_iters):
            s = base_logits + p @ phi.t()  # message to c: sum_j phi[c, j] p_j
            p = torch.sigmoid(s)
        return s


class MultiLabelMIL(BaseModel):
    def __init__(
        self,
        in_dim: int,
        n_labels: int = 6,
        d_model: int = 512,
        n_heads: int = 8,
        n_layers: int = 2,
        ffn_mult: int = 4,
        attn_dropout: float = 0.1,
        class_dropout: float = 0.0,
        dropout: float = 0.1,
        adjacency: Optional[torch.Tensor] = None,
        gcn_node_dim: int = 300,
        gcn_hidden: int = 512,
        gcn_node_init: Optional[torch.Tensor] = None,
        use_crf: bool = True,
        crf_iters: int = 2,
    ):
        super().__init__()

        self.encoder = QueryLabelMIL(
            in_dim,
            d_model,
            n_labels,
            n_heads,
            n_layers,
            ffn_mult,
            attn_dropout,
            class_dropout,
            dropout,
        )

        logging.info(
            "Initialised QLabelMIL with adjacency=%s, crf=%s and num_classes=%d",
            adjacency is not None,
            use_crf,
            n_labels,
        )

        self.use_gcn = adjacency is not None
        if self.use_gcn:
            self.gcn = LabelGCN(
                adjacency,  # type: ignore
                gcn_node_dim,
                gcn_hidden,
                d_model,
                node_init=gcn_node_init,
            )
            self.fusion_bias = nn.Parameter(torch.zeros(n_labels))

        self.crf = LabelCRFHead(n_labels, crf_iters) if use_crf else None

    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x shape: [batch_size (1), N_patches (N), input_dim (M)]
        # other: num_classes (C)

        Z, attn = self.encoder.encode(x)  # (B, C, d_model), (B, C, N)

        if self.use_gcn:
            g = self.gcn()  # (C, d)
            logits = (Z * g).sum(-1) + self.fusion_bias  # (B, C)
        else:
            logits = (Z * self.encoder.cls_weight).sum(-1) + self.encoder.cls_bias

        if self.crf is not None:
            logits = self.crf(logits)  # (B, C)

        aux_dict: Dict[str, torch.Tensor] = {"attn": attn}

        return logits, aux_dict
