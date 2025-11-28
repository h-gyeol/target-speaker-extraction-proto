# -*- coding: utf-8 -*-
"""
TS-VAD (Target Speaker Voice Activity Detection) + Adaptive Suppression

타깃 화자 임베딩(ECAPA)와 유사도로 프레임별 활성 구간을 계산한 뒤,
- 타깃 화자 활성 구간: 그대로 유지 (또는 1.2x 부스트)
- 비활성 구간: -20 dB 수준으로 강력 억제

결과물: target_boosted_final.wav
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


def ecapa_emb(model, wav, sr):
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(t).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def framewise_similarity(model, enroll_emb, wav, sr=16000, win=0.5, hop=0.25):
    """
    0.5초 창, 0.25초 hop으로 프레임별 타깃 유사도 계산.
    """
    frame_len = int(sr * win)
    hop_len = int(sr * hop)

    sims = []
    centers = []

    for start in range(0, len(wav) - frame_len, hop_len):
        frame = wav[start:start + frame_len]
        emb = ecapa_emb(model, frame, sr)
        sim = cosine(enroll_emb, emb)
        sims.append(sim)
        centers.append(start)

    sims = np.array(sims, dtype=np.float32)
    centers = np.array(centers, dtype=np.int32)
    return sims, centers


def smooth(x, k=5):
    """simple moving average smoothing."""
    if len(x) < k:
        return x
    y = np.convolve(x, np.ones(k)/k, mode="same")
    return y


def main():
    # --------------------------
    # Load audio
    # --------------------------
    mix, sr = sf.read("mixture.wav")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)

    # 5e의 출력(target_superboost.wav)이 있으면 사용, 없으면 target_supermask.wav 사용
    import os
    if os.path.exists("target_superboost.wav"):
        tgt_path = "target_superboost.wav"
        print("[INFO] Using 5e output: target_superboost.wav")
    else:
        tgt_path = "target_supermask.wav"
        print("[INFO] Using fallback: target_supermask.wav")
    
    tgt, sr2 = sf.read(tgt_path)
    if tgt.ndim > 1:
        tgt = tgt.mean(axis=1)
    if sr2 != sr:
        tgt = librosa.resample(tgt, orig_sr=sr2, target_sr=sr)

    enroll, sr3 = sf.read("enroll_target_clean.wav")
    if enroll.ndim > 1:
        enroll = enroll.mean(axis=1)
    if sr3 != sr:
        enroll = librosa.resample(enroll, orig_sr=sr3, target_sr=sr)

    # --------------------------
    # Load ECAPA model
    # --------------------------
    model = load_ecapa("cpu")
    emb_enroll = ecapa_emb(model, enroll, sr)

    # --------------------------
    # Framewise similarity
    # --------------------------
    sims, centers = framewise_similarity(model, emb_enroll, mix, sr)
    sims_s = smooth(sims, k=7)  # smoother curve

    # normalize similarities 0~1
    sims_norm = (sims_s - sims_s.min()) / (sims_s.max() - sims_s.min() + 1e-8)

    # --------------------------
    # Build time-domain mask
    # --------------------------
    mask = np.zeros_like(mix, dtype=np.float32)
    frame_len = int(sr * 0.5)
    hop_len = int(sr * 0.25)

    for sim, c in zip(sims_norm, centers):
        val = sim  # 0~1
        mask[c:c + frame_len] = np.maximum(mask[c:c + frame_len], val)

    # --------------------------
    # Adaptive suppression
    # --------------------------
    # 타깃 활성도 = mask
    # output = tgt * (1 + mask*0.3) + mix*(1-mask)*0.1

    out = (
        tgt * (1.0 + 0.3 * mask) +        # 타깃 구간: +30% 부스트
        mix * (0.1 * (1.0 - mask))        # 비타깃 구간: -20dB 수준 억제
    )

    # normalize
    peak = np.max(np.abs(out))
    if peak > 1:
        out = out / peak

    sf.write("target_boosted_final.wav", out, sr)
    print("[OK] Saved: target_boosted_final.wav")


if __name__ == "__main__":
    main()