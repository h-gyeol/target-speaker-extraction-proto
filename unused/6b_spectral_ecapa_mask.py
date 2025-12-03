# -*- coding: utf-8 -*-
"""
Spectral ECAPA Masking (타깃 임베딩 기반 주파수-시간 마스크)
Fixed version: negative similarity suppression + safe normalization
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
        run_opts={"device": device},
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
    # STFT
    # -------------------------
    S_mix = librosa.stft(mix, n_fft=512, hop_length=128, window="hann")
    S_tgt = librosa.stft(tgt, n_fft=512, hop_length=128, window="hann")

    mag_mix = np.abs(S_mix)
    pha_mix = np.angle(S_mix)

    # -------------------------
    # Frame-wise ECAPA similarity
    # -------------------------
    T = S_mix.shape[1]
    sims = np.zeros(T, dtype=np.float32)

    frame_len = int(sr * 0.2)  # 200ms
    hop = int(sr * 0.1)        # 100ms hop

    for t in range(T):
        start = t * hop
        end = start + frame_len
        if end > len(mix):
            end = len(mix)
        frame = mix[start:end]
        if len(frame) < sr * 0.1:
            continue
        emb_f = ecapa_emb(model, frame)
        sims[t] = cosine(emb_enroll, emb_f)

    # -------------------------
    # FIXED NORMALIZATION
    # -------------------------
    # 1) negative similarities = non-target → 0
    sims_p = np.maximum(sims, 0)

    # 2) if all zero, keep zero mask
    if np.max(sims_p) < 1e-6:
        sims_n = sims_p
    else:
        sims_n = sims_p / (np.max(sims_p) + 1e-8)

    # -------------------------
    # Spectral mask
    # -------------------------
    p = 1.8  # mask sharpening
    M = np.tile(sims_n[np.newaxis, :], (mag_mix.shape[0], 1))
    M = M ** p

    M = np.clip(M, 0.05, 1.0)

    # apply
    Y = M * mag_mix * np.exp(1j * pha_mix)
    y = librosa.istft(Y, hop_length=128, window="hann")

    # normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y = y / peak

    sf.write("target_spectral_boosted.wav", y, sr)
    print("[OK] Saved target_spectral_boosted.wav")


if __name__ == "__main__":
    main()