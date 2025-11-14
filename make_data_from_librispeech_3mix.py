# -*- coding: utf-8 -*-
import os, random
from pathlib import Path
import torch, torchaudio
import soundfile as sf
import numpy as np

ROOT = "data_librispeech"
OUT_MIX = "mixture.wav"
OUT_ENROLL = "enroll_target_clean.wav"
SR = 16000
MIX_SEC = 10.0
ENROLL_SEC = 20.0
SEED = 1337

def to_mono_16k(wav: torch.Tensor, sr: int) -> torch.Tensor:
    # wav: [C, T] or [T]
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

def scale_to_snr(target: np.ndarray, interferers: list, snrs_db: list):
    """
    target: 타깃 신호 (레퍼런스 0 dB)
    interferers: 간섭 신호 리스트
    snrs_db: 타깃 대비 각 간섭의 SNR(dB), 예) [-5, -8]
    """
    t_rms = rms(target)
    outs = []
    for x, snr in zip(interferers, snrs_db):
        g = (t_rms / (rms(x) + 1e-12)) * (10.0 ** (-snr / 20.0))
        outs.append(x * g)
    return [target] + outs

def pick_three_speakers(ds):
    # 화자ID -> 인덱스 리스트
    speakers = {}
    for i in range(len(ds)):
        _, sr, _, spk_id, _, _, _ = ds._walker[i], None, None, None, None, None, None
    # 위 줄은 torchaudio 내부 접근이 버전마다 달라서 안전하게 loop로 실제 로드
    for i in range(len(ds)):
        wav, sr, utt, spk, chapter, utt_id = ds[i]  # torchaudio 2.x: returns (waveform, sample_rate, utterance, speaker_id, chapter_id, utterance_id)
        speakers.setdefault(spk, []).append(i)
    spk_ids = [k for k, v in speakers.items() if len(v) >= 2]  # 등록/혼합용으로 최소 2개 필요
    assert len(spk_ids) >= 3, "3명 이상의 화자가 필요합니다(test-clean 자동 다운로드가 끝났는지 확인)."
    return random.sample(spk_ids, 3), speakers

def load_ds():
    ds = torchaudio.datasets.LIBRISPEECH(ROOT, url="test-clean", download=True)
    return ds

def main():
    random.seed(SEED)
    torch.manual_seed(SEED)

    ds = load_ds()
    spk_ids, speakers = pick_three_speakers(ds)
    tgt_spk, if1_spk, if2_spk = spk_ids
    print(f"[OK] Target speaker: {tgt_spk}, Interferers: {if1_spk}, {if2_spk}")

    # ── 타깃 등록용 20초 추출
    # 타깃 화자의 발화 중 2개 이상에서 이어 붙여 20초 확보(짧으면 패딩)
    tgt_indices = random.sample(speakers[tgt_spk], k=min(3, len(speakers[tgt_spk])))
    enroll_wavs = []
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

    # ── 혼합용 10초(서로 다른 발화 3개)
    def pick_from(spk):
        idx = random.choice(speakers[spk])
        wav, sr, *_ = ds[idx]
        return cut_or_pad(to_mono_16k(wav, sr), MIX_SEC)

    t = pick_from(tgt_spk).numpy().astype(np.float32)
    i1 = pick_from(if1_spk).numpy().astype(np.float32)
    i2 = pick_from(if2_spk).numpy().astype(np.float32)

    # 레벨 정렬: 타깃 0dB, 간섭 -5dB, -8dB
    t, i1_s, i2_s = scale_to_snr(t, [i1, i2], snrs_db=[-5.0, -8.0])

    mix = t + i1_s + i2_s
    # 안전 클리핑
    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)

    sf.write(OUT_MIX, mix, SR)
    print(f"Saved: {OUT_ENROLL} ({ENROLL_SEC:.0f}s), {OUT_MIX} ({MIX_SEC:.0f}s)")

if __name__ == "__main__":
    main()
