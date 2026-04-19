import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, MLP, global_mean_pool, AttentionalAggregation
from torch_geometric.utils import degree
from torch_scatter import scatter_softmax

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x.div(keep_prob) * torch.empty(shape, device=x.device).bernoulli_(keep_prob)

class GatedResidual(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), 1e-5)) # Slightly non-zero for gradient flow

    def forward(self, x, res):
        return res + self.gamma * x

class OptimizedNexusBlock(nn.Module):
    def __init__(self, dim, heads, drop, drop_path):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5

        # 1. Local Message Passing (Structural)
        self.norm1 = nn.LayerNorm(dim)
        self.conv = GINEConv(MLP([dim, dim * 2, dim], act="gelu", norm="layer_norm"), train_eps=True)
        self.res1 = GatedResidual(dim)

        # 2. Node -> Latent (Refinement)
        self.norm2 = nn.LayerNorm(dim)
        self.v_norm = nn.LayerNorm(dim)
        self.token_attn = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=drop)
        self.res2 = GatedResidual(dim)

        # 3. Latent -> Node (Efficient Broadcast)
        self.norm3 = nn.LayerNorm(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.kv_proj = nn.Linear(dim, dim * 2)
        self.out_proj = nn.Linear(dim, dim)
        self.res3 = GatedResidual(dim)

        # 4. FFN
        self.norm4 = nn.LayerNorm(dim)
        self.ffn = MLP([dim, dim * 4, dim], act="gelu", dropout=drop)
        self.drop_path = DropPath(drop_path)

    def forward(self, x, edge_index, edge_attr, v, batch):
        # --- Local Update ---
        h = self.conv(self.norm1(x), edge_index, edge_attr)
        x = self.res1(self.drop_path(h), x)

        # --- Latent Refinement (Node-to-Token Summary) ---
        # We use mean pool per graph as a 'virtual node' for the tokens to attend to
        v_res = v
        v = self.v_norm(v)
        node_summary = global_mean_pool(x, batch).unsqueeze(1) 
        v_context = torch.cat([v, node_summary], dim=1)
        v_upd, _ = self.token_attn(v, v_context, v_context)
        v = self.res2(self.drop_path(v_upd), v_res)

        # --- Efficient Broadcast (Token-to-Node) ---
        # Logic: Nodes only attend to the tokens of their own graph
        x_norm = self.norm3(x)
        q = self.q_proj(x_norm).view(-1, self.heads, self.head_dim) # [N, H, D]
        
        kv = self.kv_proj(v).view(v.size(0), v.size(1), 2, self.heads, self.head_dim)
        k, val = kv.unbind(2) # k, val: [B, Tokens, H, D]

        # Map tokens back to node indices without BxN matrices
        # k_expanded: [N, Tokens, H, D]
        k_exp = k[batch] 
        val_exp = val[batch]

        # Scaled Dot-Product Attention (Memory Efficient)
        # Score: [N, H, Tokens]
        scores = torch.einsum('nhd,nthd->nht', q, k_exp) * self.scale
        attn = scatter_softmax(scores, torch.zeros(1, device=x.device), dim=-1) # Local softmax over tokens
        
        # Out: [N, H, D]
        attn_out = torch.einsum('nht,nthd->nhd', attn, val_exp).reshape(-1, self.dim)
        x = self.res3(self.drop_path(self.out_proj(attn_out)), x)

        # --- Feed Forward ---
        x = x + self.drop_path(self.ffn(self.norm4(x)))
        return x, v

class CancerDetectionOmni_V18_ELITE(nn.Module):
    def __init__(self, node_in, edge_in, hidden=256, depth=12, heads=8, drop=0.1):
        super().__init__()
        self.num_tokens = 16

        self.node_enc = nn.Linear(node_in, hidden)
        self.edge_enc = nn.Linear(edge_in, hidden)
        self.deg_emb = nn.Embedding(512, hidden)

        # Xavier/Kaiming-aware initialization
        self.v_tokens = nn.Parameter(torch.randn(1, self.num_tokens, hidden) * 0.02)

        self.blocks = nn.ModuleList([
            OptimizedNexusBlock(hidden, heads, drop, (i/depth) * 0.3) 
            for i in range(depth)
        ])

        self.node_readout = AttentionalAggregation(MLP([hidden, hidden, 1]))
        
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        num_graphs = batch.max().item() + 1

        # Input Embedding
        deg = degree(edge_index[0], x.size(0)).long().clamp(max=511)
        x = self.node_enc(x) + self.deg_emb(deg)
        edge_attr = self.edge_enc(edge_attr)
        
        v = self.v_tokens.expand(num_graphs, -1, -1)

        for block in self.blocks:
            x, v = block(x, edge_index, edge_attr, v, batch)

        # Global Fusion
        node_glob = self.node_readout(x, batch)
        token_glob = v.mean(dim=1)
        
        return self.head(torch.cat([node_glob, token_glob], dim=-1))
