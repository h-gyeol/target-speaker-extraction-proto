# -*- coding: utf-8 -*-
# SepFormer 3-mix 분리 + 안전 저장(형상 강제) + 샘플레이트 자동 감지 저장

import os
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"   # symlink 대신 copy
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"       # 경고 숨김

import time
import numpy as np
import soundfile as sf
from pathlib import Path
from speechbrain.inference import SepformerSeparation as Separator

MIX_PATH = "mixture.wav"
SAVEDIR  = "pretrained_sepformer_3mix"

def to_sources_2d(x_np: np.ndarray) -> np.ndarray:
    """
    출력 shape를 [N, T] (N: 소스/화자 수, T: 샘플 길이)로 강제.
    허용 입력: [T], [N,T], [T,N], [1,N,T], [B,N,T]
    """
    x_np = np.asarray(x_np)

    # 1) 차원 정리
    if x_np.ndim == 1:                  # [T] -> [1, T]
        x_np = x_np[None, :]
    elif x_np.ndim == 3:                # [1,N,T] 또는 [B,N,T]
        x_np = x_np[0]                  # 가장 앞 배치만 사용 -> [N,T] 기대

    if x_np.ndim != 2:
        raise ValueError(f"Unexpected est shape: {x_np.shape}")

    N, T = x_np.shape

    # 2) [T,N]로 온 경우 전치: 보통 T는 수만~수십만, N은 1~4
    if N > T and T <= 8:
        x_np = x_np.T
        N, T = x_np.shape

    # 3) 방어적 처리: 여전히 앞축이 비정상적으로 크면 전치 재시도
    if N > 16 and T < 16384:
        x_np = x_np.T

    # 최종 체크
    if x_np.ndim != 2:
        raise ValueError(f"Could not coerce to [N,T], got {x_np.shape}")
    return x_np

def main():
    if not Path(MIX_PATH).exists():
        raise FileNotFoundError(MIX_PATH)

    print("[INFO] loading sepformer 3mix...")
    t0 = time.time()
    separator = Separator.from_hparams(
        source="speechbrain/sepformer-wsj03mix",
        savedir=SAVEDIR
    )
    print(f"[OK] model ready in {time.time()-t0:.2f}s")

    print("[INFO] start separation (SepFormer works at 8 kHz)...")
    t1 = time.time()
    est = separator.separate_file(path=MIX_PATH)   # torch.Tensor
    print(f"[OK] separation call returned in {time.time()-t1:.2f}s")

    # torch -> numpy
    est_np = est.detach().cpu().numpy()
    est_np = to_sources_2d(est_np)                 # → [N, T]
    print(f"[DEBUG] est shape(after coerce): {est_np.shape}")  # 예: (3, 80000)

    # === 샘플레이트 자동 감지 저장 ===
    mix_y, mix_sr = sf.read(MIX_PATH)
    est_len = est_np.shape[1]
    mix_len = len(mix_y) if mix_y.ndim == 1 else mix_y.shape[0]

    # est 길이가 mix의 약 절반이면(±2%) 8 kHz로 저장 (입력 16k → 추론 8k 패턴)
    if abs(est_len * 2 - mix_len) <= int(0.02 * mix_len):
        sr_out = max(8000, mix_sr // 2)  # 보수적으로 8k
    else:
        sr_out = mix_sr

    out_paths = []
    N = int(est_np.shape[0])                       # 소스 개수만큼 저장
    for i in range(N):
        y = est_np[i]
        if y.ndim > 1:
            y = y.mean(axis=-1)                    # 다채널이면 모노화
        y = y.astype(np.float32)
        out = f"sep_src{i}.wav"
        sf.write(out, y, sr_out)
        out_paths.append(out)

    print(f"Saved (sr={sr_out}):", out_paths)

if __name__ == "__main__":
    main()
