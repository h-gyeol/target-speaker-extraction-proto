# -*- coding: utf-8 -*-
"""
ECAPA Gain-based Spectral Masking (타깃 음성은 키우고, 방해 화자는 줄이는 방식)
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
    mag_mix = np.abs(S_mix)
    pha_mix = np.angle(S_mix)

    # -------------------------
    # Frame-wise similarity
    # -------------------------
    frame_len = int(sr * 0.2)  # 200ms
    hop = int(sr * 0.1)

    sims = []
    for t in range(0, len(mix), hop):
        frame = mix[t:t + frame_len]
        if len(frame) < sr * 0.1:
            sims.append(0.0)
            continue
        emb_f = ecapa_emb(model, frame)
        sims.append(cosine(emb_enroll, emb_f))

    sims = np.array(sims)
    sims_n = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # Frame 개수 → STFT frame 개수 보정
    T = mag_mix.shape[1]
    sims_resized = np.interp(np.linspace(0, len(sims_n) - 1, T),
                             np.arange(len(sims_n)), sims_n)

    # -------------------------
    # Gain mask: boost target, suppress others
    # -------------------------
    boost_max = 2.0      # +6dB
    suppress_min = 0.25  # -12dB

    M = suppress_min + (boost_max - suppress_min) * (sims_resized ** 1.5)
    M = np.tile(M[np.newaxis, :], (mag_mix.shape[0], 1))

    # -------------------------
    # Apply mask
    # -------------------------
    Y = M * mag_mix * np.exp(1j * pha_mix)
    y = librosa.istft(Y, hop_length=128, window="hann")

    # Normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y /= peak

    sf.write("target_gain_mask.wav", y, sr)
    print("[OK] Saved target_gain_mask.wav")

if __name__ == "__main__":
    main()
