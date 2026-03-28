#!/usr/bin/env python3
"""Verify: reverting cache size to 100_000 fixes the streaming encoder."""

import numpy as np
import mlx.core as mx
import sys
sys.path.insert(0, "/tmp/voxmlx")

from voxmlx import load_model
from voxmlx.audio import log_mel_spectrogram, pad_audio
from voxmlx.cache import RotatingKVCache


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

    # Full encode (ground truth)
    full = model.encode(mel)
    mx.eval(full)

    # Test different cache sizes
    for cache_size in [750, 2000, 10000, 100000]:
        orig_sw = model.encoder.sliding_window
        model.encoder.sliding_window = cache_size

        T = mel.shape[1]
        mel_inc = mel[:, 1:] if T % 2 != 0 else mel
        conv1_tail = conv2_tail = encoder_cache = ds_buf = None
        embeds_list = []
        chunk_size = 8

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
        inc = mx.concatenate(embeds_list, axis=0)
        mx.eval(inc)

        min_len = min(full.shape[0], inc.shape[0])
        max_diffs = []
        for i in range(min_len):
            d = mx.abs(full[i] - inc[i]).max().item()
            max_diffs.append(d)

        diffs = np.array(max_diffs)
        print(f"  cache_size={cache_size:>6d}: "
              f"max_diff={diffs.max():.6f} mean_diff={diffs.mean():.6f} "
              f"n_positions_with_diff>0.1: {(diffs > 0.1).sum()}/{min_len}")


if __name__ == "__main__":
    main()
