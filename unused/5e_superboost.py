# -*- coding: utf-8 -*-
"""
Supermask 결과물에 타깃 발화 강화(Adaptive Gain Boosting) 적용
"""

import numpy as np
import soundfile as sf
import librosa

M_PATH = "target_supermask.wav"
M_MASK_NPY = "supermask.npy"   # 5d_supermask_refine.py에서 저장하도록 해야 함

def main():
    # -----------------
    # Load enhanced target
    # -----------------
    y, sr = sf.read(M_PATH)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # -----------------
    # Load mask M
    # -----------------
    if not (os.path.exists(M_MASK_NPY)):
        print("[ERROR] supermask.npy not found")
        return

    M = np.load(M_MASK_NPY)   # shape: [freq, time]

    # -----------------
    # Compute boosting gain
    # -----------------
    # Time-domain gain vector: frame-wise median of mask
    M_time = np.median(M, axis=0)    # shape: (T,)

    # Normalize
    M_time = np.clip(M_time, 0.05, 1.0)

    # Gain rule: boost small-M regions stronger
    gain = 1.0 / (M_time + 0.20)
    gain = np.clip(gain, 1.0, 6.0)   # 최대 6배까지

    # Expand gain to full signal length
    hop = 128
    frame_len = hop * len(gain)
    gain_full = np.repeat(gain, hop)[:len(y)]

    # -----------------
    # Apply gain
    # -----------------
    y_boost = y * gain_full.astype(np.float32)

    # Normalize to prevent clipping
    peak = np.max(np.abs(y_boost))
    if peak > 1:
        y_boost /= peak

    sf.write("target_superboost.wav", y_boost, sr)
    print("[OK] Saved: target_superboost.wav")

if __name__ == "__main__":
    import os
    main()