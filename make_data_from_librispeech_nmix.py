# -*- coding: utf-8 -*-
import argparse, random
from pathlib import Path
from typing import List  # ✅ Python 3.8 호환
import numpy as np
import torch, torchaudio, soundfile as sf

ROOT = "data_librispeech"
OUT_MIX = "mixture.wav"
OUT_ENROLL = "enroll_target_clean.wav"
SR = 16000
MIX_SEC = 10.0
ENROLL_SEC = 20.0
SEED = 1337

def to_mono_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)       # [1, T]
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SR:
        wav = torchaudio.functional.resample(wav, sr, SR)
    return wav.squeeze(0)             # [T]

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
    """타깃 RMS를 기준(0 dB)으로 하고, 각 간섭을 주어진 SNR(dB)로 스케일."""
    t_rms = rms(target)
    outs: List[np.ndarray] = []
    for x, snr in zip(interfs, snrs_db):
        g = (t_rms / (rms(x) + 1e-12)) * (10.0 ** (-snr / 20.0))
        outs.append(x * g)
    return [target] + outs

def load_ds():
    return torchaudio.datasets.LIBRISPEECH(ROOT, url="test-clean", download=True)

def build_speaker_index(ds):
    speakers = {}
    for i in range(len(ds)):
        item = ds[i]
        wav, sr, *_ = item
        # torchaudio: (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
        spk_raw = item[3]
        try:
            spk = int(spk_raw)
        except Exception:
            spk = int(str(spk_raw))
        speakers.setdefault(spk, []).append(i)
    return speakers

def pick_speakers(speakers, n: int):
    rich = [k for k, v in speakers.items() if len(v) >= 2]
    pool = rich if len(rich) >= n else list(speakers.keys())
    assert len(pool) >= n, f"{n}명 이상의 화자가 필요합니다 (test-clean 다운로드 확인)."
    return random.sample(pool, n)

def default_snr_list(k: int) -> List[float]:
    # k = 간섭자 수; 예: 4명이면 [-5, -8, -11, -14]
    base, step = -5.0, -3.0
    return [base - i * step for i in range(k)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_speakers", type=int, default=3, help="혼합에 사용할 전체 화자 수 (>=2)")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    assert args.num_speakers >= 2, "num_speakers는 최소 2 이상이어야 합니다."
    random.seed(args.seed); torch.manual_seed(args.seed)

    ds = load_ds()
    speakers = build_speaker_index(ds)
    spk_list = pick_speakers(speakers, args.num_speakers)
    tgt_spk, interferers = spk_list[0], spk_list[1:]
    print(f"[OK] Target speaker: {tgt_spk}, Interferers: {', '.join(map(str, interferers))}")

    # ── 타깃 등록용 20초 (여러 발화 이어붙이기)
    tgt_indices = random.sample(speakers[tgt_spk], k=min(3, len(speakers[tgt_spk])))
    enroll_wavs: List[torch.Tensor] = []  # ✅ 한 줄에 하나씩
    total = 0
    for idx in tgt_indices:
        wav, sr, *_ = ds[idx]
        w = to_mono_16k(wav, sr)
        enroll_wavs.append(w)
        total += w.numel()
        if total / SR >= ENROLL_SEC:
            break
    enroll = torch.cat(enroll_wavs, dim=0)
    enroll = cut_or_pad(enroll, ENROLL_SEC)
    sf.write(OUT_ENROLL, enroll.numpy().astype(np.float32), SR)

    # ── 혼합용 10초: 각 화자에서 1개 발화씩 선택
    def take_one(spk: int) -> torch.Tensor:
        idx = random.choice(speakers[spk])
        wav, sr, *_ = ds[idx]
        return cut_or_pad(to_mono_16k(wav, sr), MIX_SEC)

    target = take_one(tgt_spk).numpy().astype(np.float32)
    interfs_np: List[np.ndarray] = [take_one(s).numpy().astype(np.float32) for s in interferers]

    # 간섭자 SNR 계단식 리스트
    snrs = default_snr_list(len(interfs_np))
    sigs = scale_to_snr(target, interfs_np, snrs_db=snrs)
    mix = np.sum(np.stack(sigs, axis=0), axis=0)
    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)
    sf.write(OUT_MIX, mix, SR)

    print(f"Saved: {OUT_ENROLL} ({ENROLL_SEC:.0f}s), {OUT_MIX} ({MIX_SEC:.0f}s) | N={args.num_speakers}, SNRs={snrs}")

if __name__ == "__main__":
    main()
