#!/usr/bin/env python3
"""Test: is the divergence caused by eviction (sliding window) or by the cache mechanism itself?

Compare:
1. Full encode (no cache, no sliding window)
2. Incremental with sliding_window=750 (real config)
3. Incremental with sliding_window=100000 (no eviction)
"""

import numpy as np
import mlx.core as mx
import sys
sys.path.insert(0, "/tmp/voxmlx")

from voxmlx import load_model
from voxmlx.audio import log_mel_spectrogram, pad_audio
from voxmlx.cache import RotatingKVCache


def incremental_encode(model, mel, cache_max_size, chunk_size=8):
    T = mel.shape[1]
    if T % 2 != 0:
        mel = mel[:, 1:]

    conv1_tail = None
    conv2_tail = None
    encoder_cache = None
    ds_buf = None
    embeds_list = []

    # Monkey-patch the sliding window for cache creation
    orig_sw = model.encoder.sliding_window
    model.encoder.sliding_window = cache_max_size

    for start in range(0, mel.shape[1], chunk_size):
        end = min(start + chunk_size, mel.shape[1])
        chunk = mel[:, start:end]
        result, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
            chunk, conv1_tail, conv2_tail, encoder_cache, ds_buf
        )
        if result is not None:
            mx.eval(result)
            embeds_list.append(result)

    model.encoder.sliding_window = orig_sw

    if embeds_list:
        return mx.concatenate(embeds_list, axis=0)
    return None


def main():
    print("Loading model...")
    model, sp, config = load_model()

    sr = 16000
    duration = 30.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * (200 + 100 * t) * t).astype(np.float32)

    audio_padded = pad_audio(audio)
    mel = log_mel_spectrogram(audio_padded)
    mx.eval(mel)

    # 1. Full encode
    full = model.encode(mel)
    mx.eval(full)
    print(f"Full encode: {full.shape}")

    # 2. Incremental with real sliding window (750)
    inc_750 = incremental_encode(model, mel, 750)
    mx.eval(inc_750)
    print(f"Incremental (sw=750): {inc_750.shape}")

    # 3. Incremental with huge cache (no eviction)
    inc_huge = incremental_encode(model, mel, 100000)
    mx.eval(inc_huge)
    print(f"Incremental (sw=100000): {inc_huge.shape}")

    min_len = min(full.shape[0], inc_750.shape[0], inc_huge.shape[0])
    ds = model.downsample_factor

    print(f"\n{'pos':>5} {'frame':>6} {'full_vs_sw750':>14} {'full_vs_huge':>14} {'sw750_vs_huge':>14}")
    print("-" * 60)

    for i in range(0, min_len, 25):
        d_750 = mx.abs(full[i] - inc_750[i]).max().item()
        d_huge = mx.abs(full[i] - inc_huge[i]).max().item()
        d_both = mx.abs(inc_750[i] - inc_huge[i]).max().item()
        frame = (i + 1) * ds
        print(f"{i:5d} {frame:6d} {d_750:14.6f} {d_huge:14.6f} {d_both:14.6f}")

    # Print around the transition
    fill_pos = 750 // ds
    print(f"\nAround cache fill (pos {fill_pos}):")
    for i in range(fill_pos - 3, min(fill_pos + 5, min_len)):
        d_750 = mx.abs(full[i] - inc_750[i]).max().item()
        d_huge = mx.abs(full[i] - inc_huge[i]).max().item()
        d_both = mx.abs(inc_750[i] - inc_huge[i]).max().item()
        frame = (i + 1) * ds
        print(f"{i:5d} {frame:6d} {d_750:14.6f} {d_huge:14.6f} {d_both:14.6f}")


if __name__ == "__main__":
    main()
