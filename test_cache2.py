#!/usr/bin/env python3
"""Focused test: compare full vs incremental encoder around the cache fill point."""

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

    # Generate 30s of audio to push well past the sliding window
    sr = 16000
    duration = 30.0
    t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * (200 + 100 * t) * t).astype(np.float32)

    audio_padded = pad_audio(audio)
    mel = log_mel_spectrogram(audio_padded)
    mx.eval(mel)
    print(f"Mel shape: {mel.shape}")

    # Full encode
    full_embeds = model.encode(mel)
    mx.eval(full_embeds)
    print(f"Full encode: {full_embeds.shape}")

    # Incremental encode with different chunk sizes
    for chunk_size in [8, 2]:
        print(f"\n--- chunk_size={chunk_size} ---")
        T = mel.shape[1]
        if T % 2 != 0:
            mel_inc = mel[:, 1:]
        else:
            mel_inc = mel

        conv1_tail = None
        conv2_tail = None
        encoder_cache = None
        ds_buf = None
        inc_embeds_list = []

        for start in range(0, mel_inc.shape[1], chunk_size):
            end = min(start + chunk_size, mel_inc.shape[1])
            chunk = mel_inc[:, start:end]
            result, conv1_tail, conv2_tail, encoder_cache, ds_buf = model.encode_step(
                chunk, conv1_tail, conv2_tail, encoder_cache, ds_buf
            )
            if result is not None:
                mx.eval(result)
                inc_embeds_list.append(result)

        if inc_embeds_list:
            inc_embeds = mx.concatenate(inc_embeds_list, axis=0)
            mx.eval(inc_embeds)
        else:
            print("No output!")
            continue

        min_len = min(full_embeds.shape[0], inc_embeds.shape[0])

        # Track max_diff in windows
        sw = int(model.encoder.sliding_window)
        ds = model.downsample_factor

        # The cache fills at embed position sw/ds = 750/4 = 187
        fill_pos = sw // ds

        # Check specific ranges
        ranges = [
            (0, 10, "start"),
            (fill_pos - 5, fill_pos + 5, f"around cache fill (pos {fill_pos})"),
            (min_len - 10, min_len, "end"),
        ]

        for r_start, r_end, label in ranges:
            r_start = max(0, r_start)
            r_end = min(min_len, r_end)
            print(f"\n  {label}:")
            for i in range(r_start, r_end):
                diff = mx.abs(full_embeds[i] - inc_embeds[i])
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                conv2_frame = (i + 1) * ds
                print(f"    pos={i:4d} conv2_frame={conv2_frame:5d} "
                      f"max_diff={max_diff:10.6f} mean_diff={mean_diff:10.6f}")

        # Overall stats
        all_diffs = []
        for i in range(min_len):
            d = mx.abs(full_embeds[i] - inc_embeds[i]).max().item()
            all_diffs.append(d)

        import numpy as np_
        diffs = np_.array(all_diffs)
        print(f"\n  Overall: max={diffs.max():.6f} mean={diffs.mean():.6f} median={np_.median(diffs):.6f}")

        # Find the position where max_diff first exceeds 0.5
        large = np_.where(diffs > 0.5)[0]
        if len(large) > 0:
            print(f"  First pos with max_diff > 0.5: {large[0]} (conv2_frame={large[0]*ds+ds})")
        else:
            print(f"  No position has max_diff > 0.5")

        # Check: with chunk_size=2, conv2 outputs 1 frame -> S=1 -> _update_in_place!
        if chunk_size == 2 and encoder_cache is not None:
            c = encoder_cache[0]
            print(f"\n  Cache state: offset={c.offset}, _idx={c._idx}, "
                  f"keys_shape={c.keys.shape if c.keys is not None else None}")


if __name__ == "__main__":
    main()
