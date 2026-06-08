import logging
import math

import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange

log = logging.getLogger(__name__)

def modulate(x, shift, scale):
    """AdaLN-zero modulation"""
    return x * (1 + scale) + shift

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)"""

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean() # average over projections and time
    
class FeedForward(nn.Module):
    """FeedForward network used in Transformers"""

    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    """Scaled dot-product attention with causal masking"""

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.heads = heads
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x, causal=True):
        """
        x : (B, T, D)
        """
        x = self.norm(x)
        drop = self.dropout if self.training else 0.0
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # q, k, v: (B, heads, T, dim_head)
        q, k, v = (rearrange(t, "b t (h d) -> b h t d", h=self.heads) for t in qkv)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=drop, is_causal=causal)
        out = rearrange(out, "b h t d -> b t (h d)")
        return self.to_out(out)


class ConditionalBlock(nn.Module):
    """Transformer block with AdaLN-zero conditioning"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True)
        )

        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=-1)
        )
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Block(nn.Module):
    """Standard Transformer block"""

    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()

        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    """Standard Transformer with support for AdaLN-zero blocks"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        depth,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=Block,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )

        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(self, x, c=None):

        if hasattr(self, "input_proj"):
            x = self.input_proj(x)

        if c is not None and hasattr(self, "cond_proj"):
            c = self.cond_proj(c)

        for block in self.layers:
            x = block(x) if isinstance(block, Block) else block(x, c)
        x = self.norm(x)

        if hasattr(self, "output_proj"):
            x = self.output_proj(x)
        return x

class Embedder(nn.Module):
    def __init__(
        self,
        input_dim=10,
        smoothed_dim=10,
        emb_dim=10,
        mlp_scale=4,
    ):
        super().__init__()
        self.patch_embed = nn.Conv1d(input_dim, smoothed_dim, kernel_size=1, stride=1)
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x):
        """
        x: (B, T, D)
        """
        x = x.float()
        x = x.permute(0, 2, 1)
        x = self.patch_embed(x)
        x = x.permute(0, 2, 1)
        x = self.embed(x)
        return x


class MLP(nn.Module):
    """Simple MLP with optional normalization and activation"""

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim=None,
        norm_fn=nn.LayerNorm,
        act_fn=nn.GELU,
    ):
        super().__init__()
        norm_fn = norm_fn(hidden_dim) if norm_fn is not None else nn.Identity()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            norm_fn,
            act_fn(),
            nn.Linear(hidden_dim, output_dim or input_dim),
        )

    def forward(self, x):
        """
        x: (B*T, D)
        """
        return self.net(x)


class ARPredictor(nn.Module):
    """Autoregressive predictor for next-step embedding prediction."""

    def __init__(
        self,
        *,
        num_frames,
        depth,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            depth,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


def sinusoidal_timestep_embedding(num_loops, dim):
    """Sinusoidal embedding table for loop indices k in {0, ..., num_loops-1}.

    Returns a (num_loops, dim) tensor. Row 0 is shifted to all-zeros so that the
    k=0 loop adds nothing, which makes a K=1 looped predictor reduce exactly to a
    single standard block (see LoopedTransformer / the K=1 sanity check). Each
    loop index still receives a distinct embedding.
    """
    pos = torch.arange(num_loops, dtype=torch.float32).unsqueeze(1)  # (K, 1)
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(num_loops, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe - pe[0:1]  # τ_0 = 0


class LoopedTransformer(nn.Module):
    """Weight-tied transformer: a single block applied ``num_loops`` times.

    Mirrors ``Transformer``'s scaffolding (input/cond/output projections and a
    final LayerNorm) so it is weight-comparable to a standard depth-1 transformer,
    but reuses one shared block. Uses input injection (Yang et al., ICLR 2024):
    ``h_0 = 0; h_{k+1} = block(h_k + x + τ_k, c)`` where ``x`` is the injected
    input and ``τ_k`` is the loop-k timestep embedding.
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        output_dim,
        num_loops,
        heads,
        dim_head,
        mlp_dim,
        dropout=0.0,
        block_class=ConditionalBlock,
        checkpoint=False,
    ):
        super().__init__()
        self.num_loops = num_loops
        self.checkpoint = checkpoint
        self.norm = nn.LayerNorm(hidden_dim)

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        # single weight-tied block, applied num_loops times
        self.block = block_class(hidden_dim, heads, dim_head, mlp_dim, dropout)

        self.register_buffer(
            "loop_emb", sinusoidal_timestep_embedding(num_loops, hidden_dim)
        )

    def forward(self, x, c=None):
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)

        h = torch.zeros_like(x)

        def run_block(h_val, x_val, loop_emb_val, c_val):
            inp = h_val + x_val + loop_emb_val
            if isinstance(self.block, Block):
                return self.block(inp)
            else:
                return self.block(inp, c_val)

        for k in range(self.num_loops):
            if self.checkpoint and self.training:
                # use_reentrant=False is safer, faster, and standard in torch 2.x
                h = torch.utils.checkpoint.checkpoint(
                    run_block, h, x, self.loop_emb[k], c, use_reentrant=False
                )
            else:
                inp = h + x + self.loop_emb[k]  # input injection + loop timestep emb
                h = self.block(inp) if isinstance(self.block, Block) else self.block(inp, c)

        h = self.norm(h)
        h = self.output_proj(h)
        return h


class LoopedPredictor(nn.Module):
    """Looped (weight-tied) drop-in replacement for ``ARPredictor``.

    Identical interface to ``ARPredictor`` (``forward(x, c) -> (B, T, output_dim)``)
    so the JEPA wrapper, loss, training loop, and planner need zero changes. The
    standard predictor's depth-L transformer body is replaced by a single block
    looped ``num_loops`` (K) times. ``depth`` is accepted but ignored (the looped
    body has no notion of stacked depth) so it can share the standard config keys.
    """

    def __init__(
        self,
        *,
        num_frames,
        num_loops,
        heads,
        mlp_dim,
        input_dim,
        hidden_dim,
        output_dim=None,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        depth=None,
        checkpoint=False,
    ):
        super().__init__()
        self.num_loops = num_loops
        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, input_dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = LoopedTransformer(
            input_dim,
            hidden_dim,
            output_dim or input_dim,
            num_loops,
            heads,
            dim_head,
            mlp_dim,
            dropout,
            block_class=ConditionalBlock,
            checkpoint=checkpoint,
        )

    def forward(self, x, c):
        """
        x: (B, T, d)
        c: (B, T, act_dim)
        """
        T = x.size(1)
        x = x + self.pos_embedding[:, :T]
        x = self.dropout(x)
        x = self.transformer(x, c)
        return x


def build_predictor(predictor_type="standard", num_loops=6, **kwargs):
    """Factory selecting the predictor variant and logging its parameter count.

    predictor_type: "standard" -> ARPredictor (depth-L transformer)
                    "looped"   -> LoopedPredictor (single block, K=num_loops loops)
    Extra kwargs are forwarded to the chosen predictor.
    """
    checkpoint = kwargs.pop("checkpoint", False)
    if predictor_type == "standard":
        predictor = ARPredictor(**kwargs)
        detail = f"depth={kwargs.get('depth')}"
    elif predictor_type == "looped":
        kwargs.pop("depth", None)  # depth is meaningless for a weight-tied loop
        predictor = LoopedPredictor(num_loops=num_loops, checkpoint=checkpoint, **kwargs)
        detail = f"num_loops={num_loops} checkpoint={checkpoint}"
    else:
        raise ValueError(
            f"Unknown predictor_type={predictor_type!r} (expected 'standard' or 'looped')"
        )

    n_params = sum(p.numel() for p in predictor.parameters())
    log.info(f"[predictor] type={predictor_type} {detail} params={n_params:,}")
    return predictor
