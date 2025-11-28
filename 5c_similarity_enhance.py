# -*- coding: utf-8 -*-
"""
ECAPA 기반 프레임 단위 타깃 화자 강조 필터
- 입력: enroll_target_clean.wav, target_emphasized_post.wav (없으면 target_emphasized.wav 사용)
- 출력: target_emphasized_enh.wav
"""

import os
import numpy as np
import soundfile as sf

import torch
from speechbrain.pretrained import EncoderClassifier


def load_wav_mono(path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    return y, sr


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    dot = float(np.dot(a, b))
    na = float(np.linalg.norm(a) + 1e-8)
    nb = float(np.linalg.norm(b) + 1e-8)
    return dot / (na * nb)


def get_ecapa_model(device="cpu"):
    print("[INFO] Loading ECAPA-TDNN model...")
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device},
    )
    print("[OK] ECAPA loaded.")
    return model


def ecapa_embed_waveform(model, wav: np.ndarray, sr: int, target_sr: int = 16000) -> np.ndarray:
    """
    wav: 1D float32 numpy
    sr:  original sampling rate
    """
    # 리샘플 (간단히 정수 배만 고려 / LibriSpeech라 sr=16k라서 대부분 그대로)
    if sr != target_sr:
        ratio = sr / target_sr
        idx = np.arange(0, len(wav), ratio)
        idx = idx[idx < len(wav)].astype(np.int64)
        wav = wav[idx]

    wav_t = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, T)
    with torch.no_grad():
        emb = model.encode_batch(wav_t).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def frame_indices(T: int, frame_len: int, hop_len: int):
    """신호 길이 T에 대해 (start, end) 인덱스 리스트 생성."""
    starts = list(range(0, max(T - frame_len, 1), hop_len))
    if len(starts) == 0:
        starts = [0]
    idx = []
    for s in starts:
        e = min(s + frame_len, T)
        idx.append((s, e))
        if e >= T:
            break
    return idx


def enhance_with_ecapa(enroll_path: str,
                       in_path: str,
                       out_path: str,
                       frame_sec: float = 0.5,
                       hop_sec: float = 0.25,
                       base_gain: float = 1.0,
                       alpha: float = 1.2,
                       min_gain: float = 0.4,
                       max_gain: float = 2.0,
                       device: str = "cpu"):
    """
    ECAPA 프레임 단위 유사도 기반 타깃 강조
    - frame_sec: 프레임 길이 (초)
    - hop_sec:   hop 길이 (초)
    - base_gain: 기본 게인
    - alpha:     유사도 편차에 곱해주는 계수(강도)
    - min_gain/max_gain: 게인 클리핑 범위
    """
    # 1) 입력 로드
    print(f"[INFO] Loading enroll: {enroll_path}")
    enroll_wav, enroll_sr = load_wav_mono(enroll_path)

    print(f"[INFO] Loading input: {in_path}")
    x, sr = load_wav_mono(in_path)

    # 2) ECAPA 모델 로드 + 타깃 임베딩
    model = get_ecapa_model(device=device)
    e_tgt = ecapa_embed_waveform(model, enroll_wav, enroll_sr)

    # 3) 프레임 분할
    frame_len = int(frame_sec * sr)
    hop_len = int(hop_sec * sr)
    if frame_len < 1:
        frame_len = 1
    if hop_len < 1:
        hop_len = 1

    idx_list = frame_indices(len(x), frame_len, hop_len)
    print(f"[INFO] Total frames: {len(idx_list)} (frame={frame_sec:.3f}s, hop={hop_sec:.3f}s)")

    sims = []
    frame_embs = []

    # 4) 각 프레임별 ECAPA 임베딩 & 유사도
    for (s, e) in idx_list:
        frame = x[s:e]
        if len(frame) < int(0.15 * sr):  # 150ms 미만은 스킵
            sims.append(0.0)
            frame_embs.append(None)
            continue
        emb = ecapa_embed_waveform(model, frame, sr)
        frame_embs.append(emb)
        sim = cosine_sim(e_tgt, emb)
        sims.append(sim)

    sims_np = np.array(sims, dtype=np.float32)
    print(f"[INFO] Raw similarity stats: min={sims_np.min():.3f}, max={sims_np.max():.3f}, mean={sims_np.mean():.3f}")

    # 5) 유사도 정규화 → [0,1]
    s_min = float(sims_np.min())
    s_max = float(sims_np.max())
    if s_max - s_min < 1e-4:
        norm_s = np.ones_like(sims_np) * 0.5
    else:
        norm_s = (sims_np - s_min) / (s_max - s_min)

    # 6) 게인 맵 생성
    #    norm_s ~ 0 → base_gain - alpha*0.5
    #    norm_s ~ 1 → base_gain + alpha*0.5
    gains = base_gain + alpha * (norm_s - 0.5)
    gains = np.clip(gains, min_gain, max_gain)
    print(f"[INFO] Gain stats: min={gains.min():.3f}, max={gains.max():.3f}, mean={gains.mean():.3f}")

    # 7) Overlap-Add 적용
    out = np.zeros_like(x, dtype=np.float32)
    weight = np.zeros_like(x, dtype=np.float32)

    # 간단한 해닝 윈도우 사용 (부드럽게 이어붙이기)
    win = np.hanning(frame_len).astype(np.float32)
    if frame_len == len(x):  # 아주 짧은 경우
        win = np.ones_like(x, dtype=np.float32)

    for (g, (s, e)) in zip(gains, idx_list):
        frame = x[s:e].copy()
        L = e - s
        if L <= 0:
            continue

        if L != frame_len:
            # 마지막 프레임 등 길이가 다른 경우 윈도우/프레임 잘라서 적용
            w = win[:L]
        else:
            w = win

        frame_enh = frame * g * w
        out[s:e] += frame_enh
        weight[s:e] += w

    # 8) weight로 나누어 보정
    valid = weight > 1e-6
    out[valid] /= weight[valid]

    # 9) peak normalize
    peak = float(np.max(np.abs(out)) + 1e-8)
    if peak > 1.0:
        out = out / peak

    sf.write(out_path, out, sr)
    print(f"[OK] Saved enhanced file: {out_path}")


if __name__ == "__main__":
    # 기본 입력 파일 이름 지정
    enroll_path = "enroll_target_clean.wav"
    in_path = "target_emphasized.wav"
    out_path = "target_emphasized_enh.wav"

    enhance_with_ecapa(
        enroll_path=enroll_path,
        in_path=in_path,
        out_path=out_path,
        frame_sec=0.5,
        hop_sec=0.25,
        base_gain=1.0,
        alpha=1.2,
        min_gain=0.4,
        max_gain=2.0,
        device="cpu",   # GPU 있으면 "cuda"로 바꿔도 됨
    )
