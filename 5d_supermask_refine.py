# -*- coding: utf-8 -*-
"""
ECAPA 기반 타깃 화자 Supermask Refinement (길이/샘플레이트 자동 맞춤 버전)
"""

import numpy as np
import soundfile as sf
import glob
import torch
import librosa
from speechbrain.inference import EncoderClassifier

def cosine(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def load_ecapa(device="cpu"):
    print("[INFO] Loading ECAPA model...")
    return EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )


def ecapa_embed(model, wav, sr=16000):
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(t).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def stft(y, sr, n_fft=512, hop=128):
    return librosa.stft(y, n_fft=n_fft, hop_length=hop, window="hann")


def istft(Y, hop=128):
    return librosa.istft(Y, hop_length=hop, window="hann")


def fix_sr_and_len(y, sr_in, sr_out, target_len):
    """8kHz sep_src → 16kHz mix length로 자동 보정"""
    if sr_in != sr_out:
        y = librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)
    if len(y) < target_len:
        # pad
        pad = target_len - len(y)
        y = np.pad(y, (0, pad))
    else:
        # trim
        y = y[:target_len]
    return y.astype(np.float32)


def main():

    # -------------------------------
    # Load mixture (16kHz)
    # -------------------------------
    mix, sr_mix = sf.read("mixture.wav")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)
    target_len = len(mix)

    # -------------------------------
    # Load enroll
    # -------------------------------
    enroll, sr_en = sf.read("enroll_target_clean.wav")
    if enroll.ndim > 1:
        enroll = enroll.mean(axis=1)

    # -------------------------------
    # Load separated sources
    # -------------------------------
    sep_paths = sorted(glob.glob("sep_src*.wav"))
    print("[INFO] separated sources:", sep_paths)

    seps = []
    sr_first = None

    for p in sep_paths:
        y, sr = sf.read(p)
        if y.ndim > 1:
            y = y.mean(axis=1)
        sr_first = sr if sr_first is None else sr_first
        seps.append(y.astype(np.float32))

    # -------------------------------
    # Fix SR & Length (8k → 16k)
    # -------------------------------
    seps_fixed = [
        fix_sr_and_len(y, sr_first, sr_mix, target_len) for y in seps
    ]

    # -------------------------------
    # ECAPA similarity to choose target src
    # -------------------------------
    model = load_ecapa("cpu")
    emb_enroll = ecapa_embed(model, enroll)

    sims = []
    for y in seps_fixed:
        emb = ecapa_embed(model, y)
        sims.append(cosine(emb_enroll, emb))

    sims = np.array(sims)
    print("[INFO] ECAPA similarity per source:", sims)

    idx = int(np.argmax(sims))
    y_tgt = seps_fixed[idx]
    others = [seps_fixed[i] for i in range(len(seps_fixed)) if i != idx]

    # -------------------------------
    # STFT (전체 길이 동일하므로 broadcasting OK)
    # -------------------------------
    S_mix = stft(mix, sr_mix)
    S_tgt = stft(y_tgt, sr_mix)

    S_oth = np.zeros_like(np.abs(S_mix))
    for y in others:
        S_oth += np.abs(stft(y, sr_mix))

    mag_mix = np.abs(S_mix)
    mag_tgt = np.abs(S_tgt)

    # -------------------------------
    # Supermask creation
    # -------------------------------
    p = 1.3
    eps = 1e-7
    M = (mag_tgt ** p) / (mag_tgt ** p + S_oth ** p + eps)
    M = np.clip(M, 0.05, 1.0)

    np.save("supermask.npy", M)

    # Apply mask
    Y = M * S_mix
    y = istft(Y)

    # normalize
    if np.max(np.abs(y)) > 1:
        y = y / np.max(np.abs(y))

    sf.write("target_supermask.wav", y, sr_mix)
    print("[OK] Saved: target_supermask.wav")


if __name__ == "__main__":
    main()