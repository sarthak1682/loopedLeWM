"""Offline sanity checks for the looped predictor (torch-only, no stable_worldmodel).

Run:  python sanity_check.py

Checks:
  #1  Parameter count: standard (depth=6) vs looped (K=6) predictor.
  #2  K=1 equivalence: a looped predictor with K=1 is identical to a single
      standard ConditionalBlock predictor (depth=1) given matched weights.
"""

import torch

from module import ARPredictor, LoopedPredictor, build_predictor

# LeWM predictor dims (config/train/model/lewm.yaml with embed_dim=192, history_size=3)
DIMS = dict(
    num_frames=3,
    input_dim=192,
    hidden_dim=192,
    output_dim=192,
    heads=16,
    mlp_dim=2048,
    dim_head=64,
    dropout=0.0,
    emb_dropout=0.0,
)


def count(m):
    return sum(p.numel() for p in m.parameters())


def check_param_counts():
    print("=" * 60)
    print("Sanity #1: parameter count (standard depth=6 vs looped K=6)")
    print("=" * 60)
    std = build_predictor("standard", depth=6, **DIMS)
    looped = build_predictor("looped", num_loops=6, **DIMS)
    n_std, n_loop = count(std), count(looped)
    print(f"  standard (depth=6) : {n_std:>12,} params")
    print(f"  looped   (K=6)     : {n_loop:>12,} params")
    print(f"  reduction          : {n_std / n_loop:.2f}x  "
          f"({100 * (1 - n_loop / n_std):.1f}% fewer)")
    print()


def check_k1_equivalence():
    print("=" * 60)
    print("Sanity #2: K=1 looped == depth-1 standard (matched weights)")
    print("=" * 60)
    torch.manual_seed(0)

    std = ARPredictor(depth=1, **DIMS).eval()
    looped = LoopedPredictor(num_loops=1, **DIMS).eval()

    # Copy shared weights: pos embedding, the single block, and the final norm.
    looped.pos_embedding.data.copy_(std.pos_embedding.data)
    looped.transformer.block.load_state_dict(std.transformer.layers[0].state_dict())
    looped.transformer.norm.load_state_dict(std.transformer.norm.state_dict())
    # input/cond/output projections are Identity here (input==hidden==output dim).

    B, T, D = 4, DIMS["num_frames"], DIMS["input_dim"]
    x = torch.randn(B, T, D)
    c = torch.randn(B, T, D)

    with torch.no_grad():
        out_std = std(x, c)
        out_loop = looped(x, c)

    max_diff = (out_std - out_loop).abs().max().item()
    ok = torch.allclose(out_std, out_loop, atol=1e-5)
    print(f"  max abs diff : {max_diff:.3e}")
    print(f"  K=1 equivalence: {'PASS' if ok else 'FAIL'}")
    print()
    assert ok, f"K=1 looped predictor does not match depth-1 standard (max diff {max_diff:.3e})"


if __name__ == "__main__":
    check_param_counts()
    check_k1_equivalence()
    print("All sanity checks passed.")
