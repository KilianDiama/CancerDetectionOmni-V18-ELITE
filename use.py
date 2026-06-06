import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import (
    GINEConv,
    MLP,
    global_mean_pool,
    global_add_pool,
)
from torch_geometric.utils import degree


# -------------------------
# Utils
# -------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.empty(
            shape, device=x.device, dtype=x.dtype
        ).bernoulli_(keep_prob)
        return x.div(keep_prob) * random_tensor


class GatedResidual(nn.Module):
    def __init__(self, dim: int, init_scale: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init_scale))

    def forward(self, x, res):
        return res + self.gamma * x


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.w2 = nn.Linear(dim, hidden_dim)
        self.w3 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.w1(x)
        x2 = self.w2(x)
        x = F.silu(x1) * x2
        x = self.dropout(x)
        x = self.w3(x)
        return x


class MultiHeadGraphReadout(nn.Module):
    """
    Multi-head attention pooling par graphe, sans padding.
    """
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Parameter(torch.randn(num_heads, dim) * 0.02)
        self.key_proj = nn.Linear(dim, dim)
        self.val_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch):
        # x: [N, D], batch: [N]
        N, D = x.size()
        num_graphs = int(batch.max().item()) + 1

        k = self.key_proj(x)          # [N, D]
        v = self.val_proj(x)          # [N, D]

        k = k.view(N, self.num_heads, self.head_dim)  # [N, H, Dh]
        v = v.view(N, self.num_heads, self.head_dim)  # [N, H, Dh]

        q = self.query.view(self.num_heads, self.head_dim)  # [H, Dh]

        scores = (k * q.unsqueeze(0)).sum(-1) * self.scale  # [N, H]

        scores_flat = scores.view(-1)                       # [N*H]
        batch_expanded = batch.unsqueeze(1).expand(-1, self.num_heads).reshape(-1)  # [N*H]

        exp_scores = scores_flat.exp()
        denom = global_add_pool(exp_scores, batch_expanded)  # [B]
        attn_flat = exp_scores / (denom[batch_expanded] + 1e-12)
        attn = attn_flat.view(N, self.num_heads)             # [N, H]
        attn = self.dropout(attn)

        v_flat = v.view(N * self.num_heads, self.head_dim)
        attn_flat = attn.view(N * self.num_heads, 1)
        weighted = v_flat * attn_flat                        # [N*H, Dh]

        batch_heads = batch.unsqueeze(1).expand(-1, self.num_heads).reshape(-1)  # [N*H]
        pooled = global_add_pool(weighted, batch_heads)      # [B, Dh]

        pooled = pooled.view(num_graphs, self.num_heads, self.head_dim)
        pooled = pooled.reshape(num_graphs, D)
        out = self.out_proj(pooled)                          # [B, D]
        return out


class RMSNorm(nn.Module):
    """
    RMSNorm simple, plus stable que LayerNorm pour des stacks profonds.
    """
    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return self.weight * x * norm


# -------------------------
# Nexus Block V23 (10/10)
# -------------------------

class NexusBlockV23(nn.Module):
    """
    Bloc nexus V23 :
    - Local GNN (GINE)
    - Node -> tokens via graph2token attention
    - Token self-attn (via MHA)
    - Tokens -> nodes via multi-head broadcast attention (VRAM-safe)
    - FFN sur nodes
    """
    def __init__(
        self,
        dim: int,
        heads: int,
        drop: float,
        drop_path: float,
        token_drop: float = 0.1,
        edge_drop: float = 0.1,
        node_drop: float = 0.1,
    ):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        self.edge_drop = edge_drop
        self.node_drop = nn.Dropout(node_drop)
        self.token_drop = nn.Dropout(token_drop)

        # 1. Local message passing
        self.norm1 = RMSNorm(dim)
        self.conv = GINEConv(
            MLP([dim, dim * 4, dim], act="gelu", norm="layer_norm"),
            train_eps=True,
        )
        self.res1 = GatedResidual(dim)

        # 2. Node -> tokens (graph2token attention)
        self.norm_tokens_in = RMSNorm(dim)
        self.graph2token_q = nn.Linear(dim, dim)
        self.graph2token_k = nn.Linear(dim, dim)
        self.graph2token_v = nn.Linear(dim, dim)
        self.res_tokens_in = GatedResidual(dim)

        # 3. Token self-attention
        self.token_self_attn = nn.MultiheadAttention(
            dim, heads, batch_first=True, dropout=drop
        )
        self.norm_tokens_self = RMSNorm(dim)
        self.res_tokens_self = GatedResidual(dim)

        # 4. Tokens -> nodes (multi-head broadcast attention, VRAM-safe)
        self.norm_nodes_attn = RMSNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.res_nodes_attn = GatedResidual(dim)

        # 5. FFN sur nodes
        self.norm_ffn = RMSNorm(dim)
        self.ffn = SwiGLU(dim, dim * 4, dropout=drop)

        self.drop_path = DropPath(drop_path)

    def _drop_edges(self, edge_index, edge_attr):
        if not self.training or self.edge_drop == 0.0:
            return edge_index, edge_attr

        row, col = edge_index
        num_edges = edge_index.size(1)
        keep_prob = 1.0 - self.edge_drop

        mask = torch.rand(num_edges, device=edge_index.device) < keep_prob
        if mask.sum() == 0:
            return edge_index, edge_attr

        edge_index = edge_index[:, mask]
        if edge_attr is not None:
            edge_attr = edge_attr[mask]

        return edge_index, edge_attr

    def forward(self, x, edge_index, edge_attr, v, batch):
        # --- 0. Edge dropout ---
        edge_index, edge_attr = self._drop_edges(edge_index, edge_attr)

        # --- 1. Local GNN update ---
        h = self.conv(self.norm1(x), edge_index, edge_attr)
        h = self.node_drop(h)
        x = self.res1(self.drop_path(h), x)

        # --- 2. Node -> tokens (graph2token attention) ---
        # v: [B, T, D], x: [N, D], batch: [N]
        v_res = v
        v_norm = self.norm_tokens_in(v)

        # Graph-level node summary as keys/values
        node_glob = global_mean_pool(x, batch)  # [B, D]
        k = self.graph2token_k(node_glob)       # [B, D]
        val = self.graph2token_v(node_glob)     # [B, D]

        # Tokens as queries
        q = self.graph2token_q(v_norm)          # [B, T, D]

        B, T, D = q.size()
        H = self.heads
        Dh = D // H

        q = q.view(B, T, H, Dh).transpose(1, 2)  # [B, H, T, Dh]
        k = k.view(B, 1, 1, Dh).expand(-1, H, T, -1)   # [B, H, T, Dh]
        val = val.view(B, 1, 1, Dh).expand(-1, H, T, -1)

        scores = (q * k).sum(-1) * self.scale      # [B, H, T]
        attn = scores.softmax(dim=-1)              # [B, H, T]
        attn = self.token_drop(attn)

        v_upd = (attn.unsqueeze(-1) * val).sum(dim=2)  # [B, H, Dh]
        v_upd = v_upd.view(B, D)                       # [B, D]
        v_upd = v_upd.unsqueeze(1).expand(-1, T, -1)   # [B, T, D]

        v = self.res_tokens_in(self.drop_path(v_upd), v_res)

        # --- 3. Token self-attention ---
        v_res2 = v
        v_norm2 = self.norm_tokens_self(v)
        v_self, _ = self.token_self_attn(v_norm2, v_norm2, v_norm2)
        v_self = self.token_drop(v_self)
        v = self.res_tokens_self(self.drop_path(v_self), v_res2)

        # --- 4. Tokens -> nodes (multi-head broadcast attention, VRAM-safe) ---
        x_norm = self.norm_nodes_attn(x)        # [N, D]
        qn = self.q_proj(x_norm)                # [N, D]

        # Graph-level token summary
        token_glob = v.mean(dim=1)              # [B, D]
        k = self.k_proj(token_glob)             # [B, D]
        val = self.v_proj(token_glob)           # [B, D]

        N = x.size(0)
        B = token_glob.size(0)
        H = self.heads
        Dh = self.head_dim

        qn = qn.view(N, H, Dh)                  # [N, H, Dh]
        k = k.view(B, H, Dh)                    # [B, H, Dh]
        val = val.view(B, H, Dh)                # [B, H, Dh]

        k_exp = k[batch]                        # [N, H, Dh]
        val_exp = val[batch]                    # [N, H, Dh]

        scores = (qn * k_exp).sum(-1) * self.scale   # [N, H]
        attn = scores.sigmoid().unsqueeze(-1)        # [N, H, 1]

        attn_out = attn * val_exp                    # [N, H, Dh]
        attn_out = attn_out.view(N, self.dim)        # [N, D]
        attn_out = self.out_proj(attn_out)
        attn_out = self.node_drop(attn_out)
        x = self.res_nodes_attn(self.drop_path(attn_out), x)

        # --- 5. FFN sur nodes ---
        ffn_out = self.ffn(self.norm_ffn(x))
        x = x + self.drop_path(ffn_out)

        return x, v


# -------------------------
# Modèle complet V23 (10/10)
# -------------------------

class CancerDetectionOmni_V23(nn.Module):
    """
    Modèle 10/10 :
    - API PyG standard: data.x, data.edge_index, data.edge_attr, data.batch
    - sortie: [num_graphs, 1]
    - stable, VRAM-safe, facile à intégrer.
    """
    def __init__(
        self,
        node_in: int,
        edge_in: int,
        hidden: int = 256,
        depth: int = 8,
        heads: int = 8,
        drop: float = 0.1,
        num_tokens: int = 16,
        max_degree: int = 512,
        token_drop: float = 0.1,
        edge_drop: float = 0.1,
        node_drop: float = 0.1,
        readout_heads: int = 4,
        seed: int | None = None,
    ):
        super().__init__()
        if seed is not None:
            torch.manual_seed(seed)

        self.num_tokens = num_tokens
        self.hidden = hidden
        self.max_degree = max_degree

        # Encodage des noeuds / arêtes / degré
        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        self.deg_emb = nn.Embedding(max_degree, hidden)

        # Tokens globaux + positional enc
        self.v_tokens = nn.Parameter(torch.randn(1, num_tokens, hidden) * 0.02)
        self.token_pos_emb = nn.Embedding(num_tokens, hidden)

        # Stack de blocs nexus
        self.blocks = nn.ModuleList([
            NexusBlockV23(
                dim=hidden,
                heads=heads,
                drop=drop,
                drop_path=(i / max(depth - 1, 1)) * 0.3,
                token_drop=token_drop,
                edge_drop=edge_drop,
                node_drop=node_drop,
            )
            for i in range(depth)
        ])

        # Readout nodes (multi-head attention pooling)
        self.node_readout = MultiHeadGraphReadout(
            hidden, num_heads=readout_heads, dropout=drop
        )

        # Fusion node_glob + token_glob avec cross-gating léger
        self.fusion_norm_nodes = RMSNorm(hidden)
        self.fusion_norm_tokens = RMSNorm(hidden)
        self.fusion_gate = nn.Linear(hidden * 2, hidden * 2)

        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, data):
        x, edge_index, edge_attr, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )
        num_graphs = int(batch.max().item()) + 1

        # Embedding d'entrée
        deg = degree(edge_index[0], x.size(0)).long().clamp(max=self.max_degree - 1)
        x = self.node_enc(x) + self.deg_emb(deg)
        edge_attr = self.edge_enc(edge_attr)

        # Tokens globaux par graphe + positional enc
        v = self.v_tokens.expand(num_graphs, self.num_tokens, self.hidden)
        pos_ids = torch.arange(self.num_tokens, device=x.device)
        v = v + self.token_pos_emb(pos_ids)[None, :, :]

        # Passage dans les blocs
        for block in self.blocks:
            x, v = block(x, edge_index, edge_attr, v, batch)

        # Readout global
        node_glob = self.node_readout(x, batch)      # [B, D]
        token_glob = v.mean(dim=1)                   # [B, D]

        node_glob = self.fusion_norm_nodes(node_glob)
        token_glob = self.fusion_norm_tokens(token_glob)

        fused = torch.cat([node_glob, token_glob], dim=-1)  # [B, 2D]

        # Cross-gating léger
        gate = torch.sigmoid(self.fusion_gate(fused))       # [B, 2D]
        fused = fused * gate

        out = self.head(fused)  # [B, 1]
        return out
