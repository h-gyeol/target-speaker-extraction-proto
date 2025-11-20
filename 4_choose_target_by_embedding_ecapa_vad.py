# -*- coding: utf-8 -*-
import numpy as np, soundfile as sf, sys, glob
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav
import os
import csv
from speechbrain.pretrained import EncoderClassifier
import torch

def append_log_row(log_csv, mixture_id, chosen_path, scores):
    """CSV 로그 파일에 한 줄을 append한다."""
    # logs 폴더 없으면 생성
    log_dir = os.path.dirname(log_csv)
    if log_dir and not os.path.isdir(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # 점수 정리
    if scores:
        sorted_scores = sorted([s for _, s in scores], reverse=True)
        top1 = sorted_scores[0]
        top2 = sorted_scores[1] if len(sorted_scores) >= 2 else 0.0
        margin = top1 - top2
    else:
        top1 = top2 = margin = 0.0

    # CSV 헤더
    fields = [
        "mixture_id",
        "chosen_path",
        "num_candidates",
        "top1_score",
        "top2_score",
        "margin",
    ]

    # 한 줄 데이터
    row = {
        "mixture_id": mixture_id,
        "chosen_path": chosen_path,
        "num_candidates": len(scores),
        "top1_score": f"{top1:.6f}",
        "top2_score": f"{top2:.6f}",
        "margin": f"{margin:.6f}",
    }

    # 기록
    new_file = not os.path.isfile(log_csv)
    with open(log_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if new_file:
            writer.writeheader()
        writer.writerow(row)

def simple_energy_vad(wav, frame_ms=30, hop_ms=15, energy_ratio=0.5):
    """
    간단한 에너지 기반 VAD.
    - wav: float32 1D numpy array
    - frame_ms: 프레임 길이 (기본 30ms)
    - hop_ms: 프레임 간격 (기본 15ms)
    - energy_ratio: 전체 평균 에너지 대비 음성으로 볼 비율
    """
    sr = 16000  # 프로젝트에서 16kHz 사용 중
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)

    energies = []
    segments = []

    # 전체 평균 에너지
    overall_energy = np.mean(wav**2) + 1e-10
    threshold = overall_energy * energy_ratio

    for start in range(0, len(wav) - frame_len, hop_len):
        frame = wav[start:start + frame_len]
        e = np.mean(frame**2)
        if e > threshold:
            energies.append((start, start + frame_len))

    # 연속된 프레임 병합
    merged = []
    if energies:
        cur_start, cur_end = energies[0]
        for s, e in energies[1:]:
            if s <= cur_end:  # 이어지는 경우
                cur_end = e
            else:
                merged.append((cur_start, cur_end))
                cur_start, cur_end = s, e
        merged.append((cur_start, cur_end))

    return merged  # [(start, end), ...]

def vad_embed(path, encoder, sr=16000):
    """
    VAD 기반 임베딩 추출:
    - 무음/잔향/노이즈 구간 제외
    - 음성 구간마다 embed_utterance → 평균
    """
    wav, file_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)  # mono
    wav = wav.astype(np.float32)

    # 리샘플 필요하면 리샘플 (프로젝트는 대부분 16k라 보통 그대로)
    if file_sr != sr:
        # 간단한 numpy 리샘플 대신 librosa나 torchaudio를 쓸 수 있지만
        # 의존성 증가를 피하기 위해 빠르게 frame-skip 방식 사용해도 OK
        ratio = file_sr / sr
        wav = wav[::int(ratio)] if ratio >= 1 else wav

    # 1) VAD로 음성 구간 찾기
    segments = simple_energy_vad(wav)
    if not segments:
        # fallback: 전체를 하나의 세그먼트로
        segments = [(0, len(wav))]

    # 2) 각 세그먼트를 임베딩으로 변환
    embs = []
    for (s, e) in segments:
        seg = wav[s:e]
        if len(seg) < sr * 0.2:  # 0.2초 이하 너무 짧으면 skip
            continue
        emb = encoder.embed_utterance(seg)
        embs.append(emb)

    # 3) 세그먼트가 모두 짧아서 걸러졌다면 fallback
    if not embs:
        embs.append(encoder.embed_utterance(wav))

    # 4) 평균 임베딩
    embs = np.stack(embs, axis=0)
    return embs.mean(axis=0)

def load_ecapa(device="cpu"):
    """
    ECAPA-TDNN (speechbrain) 임베딩 모델 로딩.
    - device="cpu" 또는 "cuda"
    """
    model = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": device}
    )
    return model

def ecapa_embed(path, ecapa_model, sr=16000):
    """
    ECAPA-TDNN 기반 화자 임베딩 추출.
    - path: wav 경로
    - ecapa_model: EncoderClassifier 객체
    - sr: target sample rate (16k)
    """
    wav, file_sr = sf.read(path)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav = wav.astype(np.float32)

    # 리샘플: ECAPA는 16k 기준
    if file_sr != sr:
        ratio = file_sr / sr
        wav = wav[::int(ratio)] if ratio >= 1 else wav

    # SpeechBrain의 ECAPA 입력: [batch, time]
    wav_tensor = torch.tensor(wav, dtype=torch.float32).unsqueeze(0)  # (1, T)

    # ECAPA 모델 forward
    emb = ecapa_model.encode_batch(wav_tensor).squeeze().cpu().numpy()
    return emb


enroll_wav = "enroll_target_clean.wav"
cands = sorted([p for p in glob.glob("sep_src*.wav")] + 
               [p for p in ["tas_src0.wav","tas_src1.wav"] if Path(p).exists()])

if not cands:
    print("[ERROR] 분리 결과 파일이 없습니다.")
    sys.exit(1)

def wav_to_emb(path, enc):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(axis=1)
    return enc.embed_utterance(preprocess_wav(y, sr))

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

enc = VoiceEncoder()
e_tgt = vad_embed(enroll_wav, enc)

ecapa_model = load_ecapa("cpu")
e_tgt_ecapa = ecapa_embed(enroll_wav, ecapa_model)

best_sim, best_path = -1.0, None
scores = []

if len(sys.argv) >= 2:
    mixture_id = sys.argv[1]
else:
    mixture_id = "default"

if len(sys.argv) >= 3:
    log_csv = sys.argv[2]
else:
    log_csv = "logs/target_selection_log.csv"

for p in cands:
    # 1) Resemblyzer VAD 임베딩
    emb_r = vad_embed(p, enc)
    s_r = cos(e_tgt, emb_r)
    # 2) ECAPA 임베딩
    emb_e = ecapa_embed(p, ecapa_model)
    s_e = cos(e_tgt_ecapa, emb_e)
    # 3) 최종 융합 스코어 (평균)
    s = (s_r + s_e) / 2.0
    print(f"{p}: resem={s_r:.3f}, ecapa={s_e:.3f}, fused={s:.3f}")
    scores.append((p, s))
    if s > best_sim:
        best_sim, best_path = s, p

y, sr = sf.read(best_path)
if y.ndim > 1: y = y.mean(axis=1)
sf.write("target_emphasized.wav", y, sr)
print(f"Selected: {best_path} (sim={best_sim:.3f}) -> target_emphasized.wav")
append_log_row(log_csv, mixture_id, best_path, scores)
print(f"[INFO] 로그 기록 완료: {log_csv}")