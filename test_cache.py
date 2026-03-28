#!/usr/bin/env python3
"""Test: diagnose streaming encoder divergence in voxmlx.

Compares full encode() vs incremental encode_step() at various cache sizes
to identify the root cause of the "silence after ~150 tokens" bug.

Run: cd /tmp/voxmlx && /Users/boris/code/mistal-experiments/.venv/bin/python test_cache.py
"""

import numpy as np
import mlx.core as mx
import sys
sys.path.insert(0, "/tmp/voxmlx")

from voxmlx.cache import RotatingKVCache


# ---------------------------------------------------------------------------
# Unit tests for RotatingKVCache
# ---------------------------------------------------------------------------

def test_rotating_cache_temporal_order():
    """Verify _update_in_place returns keys in wrong temporal order after wrap."""
    print("=" * 70)
    print("TEST 1: RotatingKVCache._update_in_place temporal order")
    print("=" * 70)

    max_size = 8
    cache = RotatingKVCache(max_size)
    bugs_found = 0

    for i in range(12):
        k = mx.ones((1, 1, 1, 4)) * (i + 1)
        v = mx.ones((1, 1, 1, 4)) * (i + 1)
        keys, values = cache.update_and_fetch(k, v)
        mx.eval(keys, values)

        buf = [int(x) for x in values[0, 0, :, 0].tolist()]
        expected_start = max(1, i + 2 - max_size)
        expected = list(range(expected_start, i + 2))
        ok = buf == expected

        if not ok:
            bugs_found += 1
        if i >= 7 or not ok:
            status = "OK" if ok else "WRONG ORDER"
            print(f"  step {i:2d}: offset={cache.offset:3d} _idx={cache._idx:3d} "
                  f"buf={buf} expected={expected} [{status}]")

    if bugs_found:
        print(f"\n  RESULT: _update_in_place has temporal order bug after wrap-around.")
        print(f"          (Not the primary bug -- see Test 3 for the real cause.)")
    else:
        print(f"\n  RESULT: No temporal order issues found.")
    print()
    return bugs_found > 0


def test_update_concat_correctness():
    """Verify _update_concat (S>1) maintains correct temporal order."""
    print("=" * 70)
    print("TEST 2: RotatingKVCache._update_concat temporal order (S=4)")
    print("=" * 70)

    max_size = 10
    cache = RotatingKVCache(max_size)
    all_ok = True

    for step in range(8):
        start_val = step * 4 + 1
        k = mx.arange(start_val, start_val + 4).reshape(1, 1, 4, 1).astype(mx.float32)
        v = mx.arange(start_val, start_val + 4).reshape(1, 1, 4, 1).astype(mx.float32)
        keys, values = cache.update_and_fetch(k, v)
        mx.eval(keys, values)

        buf = [int(x) for x in values[0, 0, :, 0].tolist()]
        n = min((step + 1) * 4, max_size)
        expected_start = max(1, (step + 1) * 4 - max_size + 1)
        expected = list(range(expected_start, (step + 1) * 4 + 1))
        ok = buf == expected

        if not ok:
            all_ok = False
        status = "OK" if ok else "WRONG"
        print(f"  step {step}: offset={cache.offset:3d} buf={buf} [{status}]")

    if all_ok:
        print(f"\n  RESULT: _update_concat maintains correct temporal order. No bug here.")
    else:
        print(f"\n  RESULT: _update_concat has a bug!")
    print()
    return not all_ok


def test_full_vs_incremental():
    """Compare full encode() vs incremental encode_step() with different cache sizes.

    This is the key test that identifies the root cause.
    """
    print("=" * 70)
    print("TEST 3: Full encode() vs incremental encode_step()")
    print("=" * 70)

    from voxmlx import load_model
    from voxmlx.audio import log_mel_spectrogram, pad_audio

    print("  Loading model...")
    model, sp, config = load_model()

    # Generate 30s synthetic audio
    sr = 16000
    duration = 30.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * (200 + 100 * t) * t).astype(np.float32)
    audio_padded = pad_audio(audio)
    mel = log_mel_spectrogram(audio_padded)
    mx.eval(mel)
    print(f"  Mel shape: {mel.shape} ({mel.shape[1] / 100:.1f}s)")

    # Full encode (ground truth -- matches what generate() uses)
    full = model.encode(mel)
    mx.eval(full)
    print(f"  Full encode: {full.shape[0]} embeddings")

    # Helper: run incremental encode with a given cache size
    def run_incremental(cache_size, chunk_size=8):
        orig_sw = model.encoder.sliding_window
        model.encoder.sliding_window = cache_size
        T = mel.shape[1]
        mel_inc = mel[:, 1:] if T % 2 != 0 else mel
        conv1_tail = conv2_tail = encoder_cache = ds_buf = None
        embeds_list = []
        for start in range(0, mel_inc.shape[1], chunk_size):
            end = min(start + chunk_size, mel_inc.shape[1])
            chunk = mel_inc[:, start:end]
            result, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
                chunk, conv1_tail, conv2_tail, encoder_cache, ds_buf
            )
            if result is not None:
                mx.eval(result)
                embeds_list.append(result)
        model.encoder.sliding_window = orig_sw
        return mx.concatenate(embeds_list, axis=0) if embeds_list else None

    # Test with different cache sizes
    ds = model.downsample_factor
    sw = int(model.encoder.sliding_window)
    fill_pos = sw // ds  # embed position where sw=750 cache fills

    print(f"\n  sliding_window={sw}, downsample_factor={ds}")
    print(f"  Cache fills at embed position {fill_pos} (conv2 frame {fill_pos * ds})")
    print()

    results = {}
    for cache_size in [750, 1500, 100_000]:
        inc = run_incremental(cache_size)
        mx.eval(inc)
        min_len = min(full.shape[0], inc.shape[0])

        max_diffs = np.array([
            mx.abs(full[i] - inc[i]).max().item() for i in range(min_len)
        ])

        # Find first position with large divergence
        large = np.where(max_diffs > 0.5)[0]
        first_large = large[0] if len(large) > 0 else None

        n_bad = (max_diffs > 0.1).sum()
        results[cache_size] = {
            "max": max_diffs.max(),
            "mean": max_diffs.mean(),
            "n_bad": n_bad,
            "total": min_len,
            "first_large": first_large,
        }

        print(f"  cache_size={cache_size:>6d}: max_diff={max_diffs.max():.4f} "
              f"mean_diff={max_diffs.mean():.4f} "
              f"bad_positions={n_bad}/{min_len} "
              f"first_diverge={'pos ' + str(first_large) if first_large is not None else 'none'}")

    # Detailed comparison around the fill point for sw=750
    print(f"\n  Detailed: cache_size=750 vs cache_size=100000 around fill point:")
    inc_750 = run_incremental(750)
    inc_big = run_incremental(100_000)
    mx.eval(inc_750, inc_big)

    min_len = min(full.shape[0], inc_750.shape[0], inc_big.shape[0])
    print(f"\n  {'pos':>5} {'frame':>6} {'full vs sw750':>14} {'full vs big':>14} {'sw750 vs big':>14}")
    print(f"  {'-'*5} {'-'*6} {'-'*14} {'-'*14} {'-'*14}")

    for i in range(max(0, fill_pos - 3), min(fill_pos + 6, min_len)):
        d1 = mx.abs(full[i] - inc_750[i]).max().item()
        d2 = mx.abs(full[i] - inc_big[i]).max().item()
        d3 = mx.abs(inc_750[i] - inc_big[i]).max().item()
        frame = (i + 1) * ds
        marker = " <-- cache fills here" if i == fill_pos else ""
        print(f"  {i:5d} {frame:6d} {d1:14.4f} {d2:14.4f} {d3:14.4f}{marker}")

    print()
    return results


def print_diagnosis(results):
    """Print final diagnosis based on test results."""
    print("=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    r750 = results.get(750, {})
    r100k = results.get(100_000, {})

    print(f"""
  FINDING: The encoder divergence is caused by the KV cache being sized
  to sliding_window=750, which forces key eviction starting at ~187 tokens.

  Evidence:
    - cache_size=750:    {r750.get('n_bad', '?')}/{r750.get('total', '?')} positions with max_diff > 0.1
    - cache_size=100000: {r100k.get('n_bad', '?')}/{r100k.get('total', '?')} positions with max_diff > 0.1

  The full encode() path (used in generate()) does NOT enforce sliding
  window attention -- it uses mask="causal" without any window limit.
  So every frame can attend to ALL prior frames. The streaming path
  with cache_size=750 evicts frames beyond the window, producing
  different encoder outputs that the decoder cannot handle.

  ROOT CAUSE: Fix #1 (changing RotatingKVCache(100_000) to
  RotatingKVCache(int(self.encoder.sliding_window))) was INCORRECT.
  The original value of 100,000 was intentional -- it prevented
  eviction, making the streaming encoder match the full encoder.
  Reducing it to 750 introduced the eviction-based divergence.

  FIX: In model.py line 122, REVERT to the original large cache size:
    RotatingKVCache(100_000)

  Or more precisely, size it to the maximum expected sequence length
  (max audio duration in conv2 frames = duration_sec * 16000 / 320).

  Fix #2 (cache.py trim calculation) is a VALID bug fix that corrects
  an off-by-one error in _update_concat. Keep it.

  ADDITIONAL (minor) BUG: RotatingKVCache._update_in_place does NOT
  reorder keys/values into temporal order after wrap-around, unlike
  _update_concat which calls _temporal_order(). This would cause
  incorrect causal masking for S=1 updates if the cache ever wraps.
  With the large cache size this is not triggered, but should be fixed
  for robustness.

  OPTIONAL IMPROVEMENT: encoder.py __call__() (full encode path) should
  also enforce sliding_window attention for correctness with long audio,
  using an explicit sliding window causal mask instead of plain "causal".
""")


if __name__ == "__main__":
    bug1 = test_rotating_cache_temporal_order()
    bug2 = test_update_concat_correctness()

    try:
        results = test_full_vs_incremental()
        print_diagnosis(results)
    except Exception as e:
        print(f"Model test failed: {e}")
        import traceback
        traceback.print_exc()

        print("\n" + "=" * 70)
        print("PARTIAL DIAGNOSIS (based on unit tests only)")
        print("=" * 70)
        print("""
  The RotatingKVCache._update_in_place path has a temporal ordering bug
  after wrap-around (keys not in chronological order in the buffer).
  This interacts badly with mask="causal" in the encoder attention.

  The primary fix should be to use a large cache (100,000) to prevent
  eviction, matching the full encode() path behavior.
""")
