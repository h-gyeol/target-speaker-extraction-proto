# -*- coding: utf-8 -*-
"""
Soft Gate + Rescue Boost
끊김 없이 타깃 화자를 전체 구간에서 유지하는 안정 버전
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
    # ------------------
    # Load audio
    # ------------------
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

    # ------------------
    # load ECAPA
    # ------------------
    model = load_ecapa("cpu")
    emb_enroll = ecapa_emb(model, enroll)

    # ------------------
    # Frame similarity
    # ------------------
    frame_len = int(sr * 0.2)
    hop = int(sr * 0.1)

    sims = []
    for t in range(0, len(mix), hop):
        frame = mix[t:t+frame_len]
        if len(frame) < frame_len//2:
            sims.append(0.0)
            continue
        emb_f = ecapa_emb(model, frame)
        sims.append(cosine(emb_enroll, emb_f))

    sims = np.array(sims)

    # normalize 0..1
    sims_n = (sims - sims.min()) / (sims.max() - sims.min() + 1e-8)

    # ------------------
    # Soft mask (core)
    # ------------------
    # similarity ↑ → gain ↑ (최대 2.5x)
    # similarity ↓ → gain ↓ but not too small (min 0.6x)
    #
    # 💡 Hard Gate와 달리 "연속적"이므로 끊김이 없음
    #
    gain_min = 0.60     # 낮은 구간도 완전 죽이지 않음
    gain_max = 2.50     # 높은 구간은 크게 boost

    gains = gain_min + (gain_max - gain_min) * (sims_n ** 1.5)

    # rescue boost (가장 중요한 부분)
    # similarity 낮아도 타깃이 아주 약하게라도 들어있다는 가정하에
    # 낮은 구간에서는 half-boost를 0.1 추가
    rescue = 0.10 * (1.0 - sims_n)
    gains += rescue

    # ------------------
    # STFT 규격화
    # ------------------
    S = librosa.stft(mix, n_fft=512, hop_length=128, window="hann")
    mag = np.abs(S)
    pha = np.angle(S)

    T = mag.shape[1]

    gains_resized = np.interp(
        np.linspace(0, len(gains) - 1, T),
        np.arange(len(gains)),
        gains
    )

    M = np.tile(gains_resized[np.newaxis, :], (mag.shape[0], 1))

    # ------------------
    # Apply mask
    # ------------------
    Y = M * mag * np.exp(1j * pha)
    y = librosa.istft(Y, hop_length=128, window="hann")

    # normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y /= peak

    sf.write("target_soft_gate_rescue.wav", y, sr)
    print("[OK] Saved target_soft_gate_rescue.wav")


if __name__ == "__main__":
    main()