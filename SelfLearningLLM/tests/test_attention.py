"""tests/test_attention.py

Phase 3 tests for the low-level building blocks: RoPE and Grouped Query
Attention. Uses small hand-picked dimensions (not a full model) so these
run in milliseconds on CPU.
"""

from __future__ import annotations

import math

import pytest
import torch

from model.attention import GroupedQueryAttention
from model.normalization import RMSNorm
from model.rope import apply_rotary_emb, precompute_rope_cache

torch.manual_seed(0)


# --------------------------------------------------------------------------- #
# RMSNorm
# --------------------------------------------------------------------------- #


def test_rmsnorm_output_has_unit_rms_before_gain():
    norm = RMSNorm(dim=16, eps=1e-8)
    torch.nn.init.constant_(norm.weight, 1.0)  # neutralize the learned gain
    x = torch.randn(4, 16) * 5.0  # arbitrary scale
    out = norm(x)
    rms = out.pow(2).mean(dim=-1).sqrt()
    assert torch.allclose(rms, torch.ones(4), atol=1e-3)


def test_rmsnorm_is_invariant_to_input_scale():
    norm = RMSNorm(dim=8, eps=1e-8)
    x = torch.randn(2, 8)
    out1 = norm(x)
    out2 = norm(x * 3.0)
    assert torch.allclose(out1, out2, atol=1e-3)


# --------------------------------------------------------------------------- #
# RoPE
# --------------------------------------------------------------------------- #


def test_rope_cache_shape():
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=32)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_rope_rejects_odd_head_dim():
    with pytest.raises(ValueError):
        precompute_rope_cache(head_dim=7, max_seq_len=16)


def test_rope_preserves_vector_norm():
    """Rotation should not change the magnitude of q/k vectors."""
    head_dim = 16
    cos, sin = precompute_rope_cache(head_dim=head_dim, max_seq_len=10)
    x = torch.randn(1, 1, 10, head_dim)
    rotated = apply_rotary_emb(x, cos, sin)
    norm_before = x.norm(dim=-1)
    norm_after = rotated.norm(dim=-1)
    assert torch.allclose(norm_before, norm_after, atol=1e-4)


def test_rope_position_zero_is_identity_like():
    """At position 0 every rotation angle is 0, so cos=1, sin=0 and the
    vector should pass through unchanged."""
    head_dim = 16
    cos, sin = precompute_rope_cache(head_dim=head_dim, max_seq_len=4)
    x = torch.randn(1, 1, 1, head_dim)
    rotated = apply_rotary_emb(x, cos[:1], sin[:1])
    assert torch.allclose(rotated, x, atol=1e-5)


def test_rope_dot_product_depends_only_on_relative_position():
    """Core RoPE property: <RoPE(q, i), RoPE(k, j)> should depend only on
    (i - j), not on i and j independently. We verify this by checking
    that shifting both positions by the same offset leaves the dot
    product unchanged.
    """
    head_dim = 16
    max_len = 20
    cos, sin = precompute_rope_cache(head_dim=head_dim, max_seq_len=max_len)

    q = torch.randn(1, 1, 1, head_dim)
    k = torch.randn(1, 1, 1, head_dim)

    def dot_at(pos_q: int, pos_k: int) -> torch.Tensor:
        rq = apply_rotary_emb(q, cos[pos_q : pos_q + 1], sin[pos_q : pos_q + 1])
        rk = apply_rotary_emb(k, cos[pos_k : pos_k + 1], sin[pos_k : pos_k + 1])
        return (rq * rk).sum()

    d1 = dot_at(5, 2)   # relative offset = 3
    d2 = dot_at(10, 7)  # relative offset = 3
    d3 = dot_at(15, 12)  # relative offset = 3
    assert torch.allclose(d1, d2, atol=1e-3)
    assert torch.allclose(d1, d3, atol=1e-3)


# --------------------------------------------------------------------------- #
# Grouped Query Attention
# --------------------------------------------------------------------------- #


def _make_attn(n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32):
    return GroupedQueryAttention(
        d_model=d_model, n_query_heads=n_query_heads, n_kv_heads=n_kv_heads, head_dim=head_dim
    )


def test_gqa_rejects_non_divisible_heads():
    with pytest.raises(ValueError):
        _make_attn(n_query_heads=5, n_kv_heads=2)


def test_gqa_output_shape():
    attn = _make_attn()
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 10, 32)
    out, present_kv = attn(x, cos[:10], sin[:10], use_cache=False)
    assert out.shape == (2, 10, 32)
    assert present_kv is None


def test_gqa_returns_cache_when_requested():
    attn = _make_attn(n_query_heads=4, n_kv_heads=2, head_dim=8)
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16)
    x = torch.randn(1, 5, 32)
    out, present_kv = attn(x, cos[:5], sin[:5], use_cache=True)
    k, v = present_kv
    assert k.shape == (1, 2, 5, 8)  # (batch, n_kv_heads, seq_len, head_dim)
    assert v.shape == (1, 2, 5, 8)


def test_gqa_is_causal_full_sequence():
    """Changing a LATER token in the input must not change an EARLIER
    token's output (causal masking correctness), when processing the
    whole sequence at once with no cache."""
    attn = _make_attn(n_query_heads=2, n_kv_heads=1, head_dim=8, d_model=16)
    attn.eval()
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16)

    x = torch.randn(1, 6, 16)
    out_a, _ = attn(x, cos[:6], sin[:6])

    x_modified = x.clone()
    x_modified[:, -1, :] = torch.randn(16)  # perturb only the LAST token
    out_b, _ = attn(x_modified, cos[:6], sin[:6])

    # every position except the last must be unaffected
    assert torch.allclose(out_a[:, :-1, :], out_b[:, :-1, :], atol=1e-5)
    # the last position (which sees itself) generally DOES change
    assert not torch.allclose(out_a[:, -1, :], out_b[:, -1, :], atol=1e-5)


def test_gqa_kv_cache_matches_full_forward():
    """Feeding a prompt token-by-token through the KV cache must produce
    IDENTICAL output (for the newest token at each step) to running the
    whole sequence through in one shot without a cache. This is the
    correctness property that makes cached generation valid, not just fast.
    """
    torch.manual_seed(42)
    attn = _make_attn(n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32)
    attn.eval()
    seq_len = 7
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=seq_len)
    x = torch.randn(1, seq_len, 32)

    # (a) one-shot, no cache
    full_out, _ = attn(x, cos, sin, use_cache=False)

    # (b) incremental, one token at a time through a growing cache
    past_kv = None
    incremental_outs = []
    for t in range(seq_len):
        step_out, past_kv = attn(
            x[:, t : t + 1, :], cos[t : t + 1], sin[t : t + 1], past_key_value=past_kv, use_cache=True
        )
        incremental_outs.append(step_out)
    incremental_out = torch.cat(incremental_outs, dim=1)

    assert torch.allclose(full_out, incremental_out, atol=1e-4)


def test_gqa_kv_cache_chunked_matches_full_forward():
    """Same correctness property as above, but feeding the cache in two
    unequal chunks rather than one token at a time -- exercises the
    explicit offset-aware mask branch (seq_len > 1 with a non-empty
    cache), not just the trivial seq_len == 1 branch.
    """
    torch.manual_seed(7)
    attn = _make_attn(n_query_heads=4, n_kv_heads=2, head_dim=8, d_model=32)
    attn.eval()
    seq_len = 9
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=seq_len)
    x = torch.randn(1, seq_len, 32)

    full_out, _ = attn(x, cos, sin, use_cache=False)

    first_chunk, second_chunk = 4, 5
    out1, past_kv = attn(x[:, :first_chunk], cos[:first_chunk], sin[:first_chunk], use_cache=True)
    out2, _ = attn(
        x[:, first_chunk:],
        cos[first_chunk : first_chunk + second_chunk],
        sin[first_chunk : first_chunk + second_chunk],
        past_key_value=past_kv,
        use_cache=True,
    )
    chunked_out = torch.cat([out1, out2], dim=1)
    assert torch.allclose(full_out, chunked_out, atol=1e-4)


def test_gqa_with_equal_heads_still_works():
    """n_query_heads == n_kv_heads degenerates to ordinary MHA; make sure
    that path (enable_gqa=False) also works correctly."""
    attn = _make_attn(n_query_heads=4, n_kv_heads=4, head_dim=8, d_model=32)
    cos, sin = precompute_rope_cache(head_dim=8, max_seq_len=16)
    x = torch.randn(2, 6, 32)
    out, _ = attn(x, cos[:6], sin[:6])
    assert out.shape == (2, 6, 32)
