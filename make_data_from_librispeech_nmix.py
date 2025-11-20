# -*- coding: utf-8 -*-
import argparse, random
from pathlib import Path
from typing import List
import numpy as np
import torch, torchaudio, soundfile as sf

ROOT = "data_librispeech"
OUT_MIX = "mixture.wav"
OUT_ENROLL = "enroll_target_clean.wav"
SR = 16000
MIX_SEC = 10.0
ENROLL_SEC = 20.0
SEED = 1337

# ============================================================
# 1) Utility
# ============================================================

def to_mono_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)

def cut_or_pad(wav: torch.Tensor, sec: float) -> torch.Tensor:
    T = int(SR * sec)
    if wav.numel() >= T:
        return wav[:T]
    out = torch.zeros(T, dtype=wav.dtype)
    out[: wav.numel()] = wav
    return out

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.clip(x, -1.0, 1.0))))) + 1e-12

def scale_to_snr(target: np.ndarray, interfs: List[np.ndarray], snrs_db: List[float]) -> List[np.ndarray]:
    t_rms = rms(target)
    outs: List[np.ndarray] = []
    for x, snr in zip(interfs, snrs_db):
        g = (t_rms / (rms(x) + 1e-12)) * (10.0 ** (-snr / 20.0))
        outs.append(x * g)
    return [target] + outs

# ============================================================
# 2) LIBRISPEECH LOAD
# ============================================================

def load_ds():
    return torchaudio.datasets.LIBRISPEECH(ROOT, url="test-clean", download=True)

def build_speaker_index(ds):
    speakers = {}
    for i in range(len(ds)):
        item = ds[i]
        spk_raw = item[3]
        try:
            spk = int(spk_raw)
        except Exception:
            spk = int(str(spk_raw))
        speakers.setdefault(spk, []).append(i)
    return speakers

def pick_speakers(speakers, n: int):
    rich = [k for k, v in speakers.items() if len(v) >= 3]
    pool = rich if len(rich) >= n else list(speakers.keys())
    assert len(pool) >= n, f"{n}명 이상의 화자가 필요합니다."
    return random.sample(pool, n)

# ============================================================
# 3) Core: build_long_audio → 12초 이상 이어붙이기
# ============================================================

def build_long_audio(ds, indices: List[int], min_sec=12.0) -> torch.Tensor:
    """여러 발화를 이어붙여 최소 min_sec 이상 길이 확보."""
    out = []
    total = 0
    for idx in indices:
        wav, sr, *_ = ds[idx]
        w = to_mono_16k(wav, sr)
        out.append(w)
        total += w.numel()
        if total / SR >= min_sec:
            break
    return torch.cat(out, dim=0)

def random_crop_10sec(wav: torch.Tensor) -> np.ndarray:
    """이어붙인 긴 음성에서 랜덤 10초를 추출"""
    T = int(MIX_SEC * SR)
    if wav.numel() <= T:
        wav = cut_or_pad(wav, MIX_SEC)
        return wav.numpy().astype(np.float32)
    max_start = wav.numel() - T
    s = random.randint(0, max_start)
    return wav[s:s+T].numpy().astype(np.float32)

def default_snr_list(k: int) -> List[float]:
    base, step = -5.0, -3.0
    return [base - i * step for i in range(k)]

# ============================================================
# 4) Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_speakers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    assert args.num_speakers >= 2
    random.seed(args.seed); torch.manual_seed(args.seed)

    ds = load_ds()
    speakers = build_speaker_index(ds)
    spk_list = pick_speakers(speakers, args.num_speakers)

    tgt_spk, interferers = spk_list[0], spk_list[1:]
    print(f"[OK] Target speaker: {tgt_spk}, Interferers: {interferers}")

    # --- ENROLL: 20초 이어붙이기 ---
    tgt_idxs = random.sample(speakers[tgt_spk], k=min(5, len(speakers[tgt_spk])))
    enroll_long = build_long_audio(ds, tgt_idxs, min_sec=20.0)
    enroll = cut_or_pad(enroll_long, ENROLL_SEC)
    sf.write(OUT_ENROLL, enroll.numpy().astype(np.float32), SR)

    # --- 각 화자별 긴 오디오 확보 (12초 이상) ---
    def long_audio_for(spk):
        idxs = random.sample(speakers[spk], k=min(6, len(speakers[spk])))
        long = build_long_audio(ds, idxs, min_sec=12.0)
        return random_crop_10sec(long)

    target = long_audio_for(tgt_spk)
    interfs_np = [long_audio_for(spk) for spk in interferers]

    # SNR scaling
    snrs = default_snr_list(len(interfs_np))
    sigs = scale_to_snr(target, interfs_np, snrs)
    mix = np.sum(np.stack(sigs, axis=0), axis=0)
    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)

    sf.write(OUT_MIX, mix, SR)
    print(f"Saved: {OUT_MIX}, N={args.num_speakers}, SNRs={snrs}")

if __name__ == "__main__":
    main()
