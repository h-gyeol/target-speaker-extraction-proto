# -*- coding: utf-8 -*-
# Iterative SepFormer separation: peel-off loop for >3 speakers
import os
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import numpy as np
import soundfile as sf
import torchaudio
from pathlib import Path
from speechbrain.inference import SepformerSeparation as Separator

MIX_PATH = "mixture.wav"
SAVEDIR  = "pretrained_sepformer_3mix"
WORK_SR  = 8000            # SepFormer wsj0-3mix는 8 kHz
MAX_ITERS = 4              # 반복 횟수 상한
MAX_SOURCES = 12           # 전체 최대 화자 수 상한
STOP_RMS_RATIO = 0.06      # 잔차/초기믹스 RMS 비가 이 값 미만이면 중단(≈ -24 dB)

def rms(x: np.ndarray) -> float:
    x = x.astype(np.float32)
    return float(np.sqrt(np.mean(np.square(x)))) + 1e-12

def resample_16k_to_8k(y16: np.ndarray, sr16: int) -> np.ndarray:
    if sr16 == WORK_SR:
        return y16.astype(np.float32)
    t = torch_from(y16)
    y8 = torchaudio.functional.resample(t, sr16, WORK_SR).numpy()
    return y8.astype(np.float32)

def torch_from(y: np.ndarray):
    import torch
    if y.ndim == 1:
        y = y[None, :]
    return torch.from_numpy(y).float().squeeze(0)

def to_sources_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 1:
        arr = arr[None, :]
    elif arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected est shape: {arr.shape}")
    N, T = arr.shape
    if N > T and T <= 8:
        arr = arr.T
    return arr

def main():
    if not Path(MIX_PATH).exists():
        raise FileNotFoundError(MIX_PATH)

    # 입력 읽고 8k로 다운샘플(모노화 포함)
    y, sr = sf.read(MIX_PATH)
    if y.ndim > 1:
        y = y.mean(axis=1)
    import torch
    y8 = resample_16k_to_8k(y, sr)
    y8 = y8.astype(np.float32)

    mix_rms0 = rms(y8)
    cur = y8.copy()
    out_idx = 0
    all_paths = []

    separator = Separator.from_hparams(source="speechbrain/sepformer-wsj03mix",
                                       savedir=SAVEDIR)

    for it in range(MAX_ITERS):
        # 임시 입력을 파일로 저장(8k) → separate_file 사용
        tmp_in = f"_iter_input_{it}.wav"
        sf.write(tmp_in, cur, WORK_SR)

        est = separator.separate_file(path=tmp_in)   # [?, ?]
        est_np = est.detach().cpu().numpy()
        est_np = to_sources_2d(est_np)               # [N, T8k]

        # 무의미한(거의 0) 소스 제거 + 저장
        used = np.zeros_like(cur)
        for i in range(est_np.shape[0]):
            src = est_np[i].astype(np.float32)
            if src.ndim > 1:
                src = src.mean(axis=-1)
            if rms(src) < 1e-4:      # 아주 미약하면 스킵
                continue
            out = f"sep_src{out_idx}.wav"
            sf.write(out, src, WORK_SR)
            all_paths.append(out)
            used[:len(src)] += src
            out_idx += 1
            if out_idx >= MAX_SOURCES:
                break

        # 정지 조건: 소스 추가 없거나 최대치
        if it == 0 and len(all_paths) == 0:
            # 첫 iter에 아무 것도 못 뽑으면 중단
            break
        if out_idx >= MAX_SOURCES:
            break

        # 잔차 계산 및 종료 판단
        L = min(len(cur), len(used))
        resid = (cur[:L] - used[:L]).astype(np.float32)
        rratio = rms(resid) / mix_rms0
        # 다음 반복 입력은 잔차
        cur = resid

        # 충분히 작으면 중단
        if rratio < STOP_RMS_RATIO:
            break

    print(f"Saved (sr={WORK_SR}):", all_paths)

if __name__ == "__main__":
    main()
