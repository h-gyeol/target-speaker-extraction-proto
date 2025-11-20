# -*- coding: utf-8 -*-
"""
Hard-Gated ECAPA Gain Masking
타깃 프레임은 키우고, 비타깃 프레임은 줄이는 방식 (가장 안정적)
"""

import numpy as np
import soundfile as sf
import torch
import librosa
from speechbrain.inference import EncoderClassifier


def load_ecapa(device="cpu"):
    print("[INFO] Loading ECAPA model...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )


def ecapa_emb(model, wav):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        e = model.encode_batch(t).squeeze().cpu().numpy()
    return e.astype(np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def main():
    # -------------------------
    # Load audio
    # -------------------------
    mix, sr = sf.read("mixture.wav")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)

    tgt, sr2 = sf.read("target_boosted_final.wav")
    if tgt.ndim > 1:
        tgt = tgt.mean(axis=1)
    tgt = librosa.resample(tgt, orig_sr=sr2, target_sr=sr)

    enroll, sr3 = sf.read("enroll_target_clean.wav")
    if enroll.ndim > 1:
        enroll = enroll.mean(axis=1)
    enroll = librosa.resample(enroll, orig_sr=sr3, target_sr=sr)

    # -------------------------
    # ECAPA load
    # -------------------------
    model = load_ecapa("cpu")
    emb_enroll = ecapa_emb(model, enroll)

    # -------------------------
    # Frame similarity
    # -------------------------
    frame_len = int(sr * 0.2)  # 200ms
    hop = int(sr * 0.1)

    sims = []
    for t in range(0, len(mix), hop):
        frame = mix[t:t + frame_len]
        if len(frame) < sr * 0.08:
            sims.append(0.0)
            continue
        emb_f = ecapa_emb(model, frame)
        sims.append(cosine(emb_enroll, emb_f))

    sims = np.array(sims)

    # Normalize 0..1
    sims_n = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # -------------------------
    # Hard Gate
    # -------------------------
    threshold = np.median(sims_n)
    print(f"[INFO] similarity threshold = {threshold:.3f}")

    target_frames = sims_n >= threshold
    suppress_frames = sims_n < threshold

    # -------------------------
    # STFT
    # -------------------------
    S = librosa.stft(mix, n_fft=512, hop_length=128, window="hann")
    mag = np.abs(S)
    pha = np.angle(S)

    T = mag.shape[1]

    # Resize sims → STFT frame count
    sims_resized = np.interp(np.linspace(0, len(sims_n) - 1, T),
                             np.arange(len(sims_n)), sims_n)

    # Hard gate version
    M = np.zeros_like(sims_resized)

    # 타깃 구간: +6dB 부스트
    M[sims_resized >= threshold] = 2.0   # gain 2x

    # 비타깃 구간: -12dB 감소
    M[sims_resized < threshold] = 0.25   # gain 0.25x

    # broadcast → frequency x time
    M = np.tile(M[np.newaxis, :], (mag.shape[0], 1))

    # -------------------------
    # Apply mask
    # -------------------------
    Y = M * mag * np.exp(1j * pha)
    y = librosa.istft(Y, hop_length=128, window="hann")

    # normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y = y / peak

    out = "target_hard_gate.wav"
    sf.write(out, y, sr)
    print("[OK] Saved:", out)


if __name__ == "__main__":
    main()