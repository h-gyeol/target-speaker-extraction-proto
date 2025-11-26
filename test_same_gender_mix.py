# -*- coding: utf-8 -*-
"""
테스트용: 동성(같은 성별) 화자 3명 혼합 데이터 생성 및 분리 테스트
- LibriSpeech test-clean에서 동성 화자만 선택
- 기존 파이프라인으로 분리 성능 확인
- Ground Truth 파일도 함께 저장하여 검증 가능
"""

import argparse
import random
import time
from pathlib import Path
from typing import List, Dict
import numpy as np
import torch
import torchaudio
import soundfile as sf

# ============================================================
# 설정
# ============================================================
ROOT = "data_librispeech"
SPEAKERS_TXT = Path(ROOT) / "LibriSpeech" / "SPEAKERS.TXT"
OUT_MIX = "mixture.wav"
OUT_ENROLL = "enroll_target_clean.wav"
OUT_GT_TARGET = "gt_target.wav"  # Ground Truth: 타겟 화자 원본
SR = 16000
MIX_SEC = 10.0
ENROLL_SEC = 20.0
DEFAULT_SEED = None  # None이면 매번 랜덤


# ============================================================
# 1) SPEAKERS.TXT 파싱 → 성별별 화자 목록
# ============================================================
def parse_speakers_txt(txt_path: Path, subset: str = "test-clean") -> Dict[str, List[int]]:
    """
    SPEAKERS.TXT를 파싱하여 성별별 화자 ID 목록 반환
    Returns: {"M": [id1, id2, ...], "F": [id1, id2, ...]}
    """
    gender_map = {"M": [], "F": []}
    
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            # 주석 또는 빈 줄 스킵
            if not line or line.startswith(";"):
                continue
            
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 3:
                continue
            
            try:
                spk_id = int(parts[0])
                gender = parts[1].upper()
                spk_subset = parts[2].strip()
                
                # 원하는 subset만 필터링
                if spk_subset == subset and gender in gender_map:
                    gender_map[gender].append(spk_id)
            except (ValueError, IndexError):
                continue
    
    return gender_map


# ============================================================
# 2) 유틸리티 함수들 (기존 코드 재사용)
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


def default_snr_list(k: int) -> List[float]:
    """방해 화자들의 SNR 리스트 생성 (타겟 대비)"""
    base, step = -5.0, -3.0
    return [base - i * step for i in range(k)]


# ============================================================
# 3) LibriSpeech 데이터 로딩
# ============================================================
def load_ds():
    return torchaudio.datasets.LIBRISPEECH(ROOT, url="test-clean", download=False)


def build_speaker_index(ds) -> Dict[int, List[int]]:
    """데이터셋에서 화자별 샘플 인덱스 매핑"""
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


def pick_speakers_by_gender(speakers: Dict[int, List[int]], 
                            gender_list: List[int], 
                            n: int) -> List[int]:
    """
    특정 성별의 화자 중 데이터셋에 존재하는 화자만 n명 선택
    """
    # 데이터셋에 실제로 존재하는 화자만 필터링
    available = [spk for spk in gender_list if spk in speakers]
    
    # 발화 수가 3개 이상인 화자 우선
    rich = [spk for spk in available if len(speakers[spk]) >= 3]
    pool = rich if len(rich) >= n else available
    
    if len(pool) < n:
        raise ValueError(f"선택 가능한 화자가 {len(pool)}명뿐입니다. {n}명 필요.")
    
    return random.sample(pool, n)


# ============================================================
# 4) 긴 오디오 생성 및 크롭
# ============================================================
def build_long_audio(ds, indices: List[int], min_sec: float = 12.0) -> torch.Tensor:
    """여러 발화를 이어붙여 최소 min_sec 이상 길이 확보"""
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
    return wav[s : s + T].numpy().astype(np.float32)


# ============================================================
# 5) 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="동성 화자 3명 혼합 테스트")
    ap.add_argument("--gender", type=str, default="M", choices=["M", "F"],
                    help="화자 성별 선택: M(남성) 또는 F(여성)")
    ap.add_argument("--num_speakers", type=int, default=3,
                    help="혼합할 화자 수 (기본값: 3)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help="랜덤 시드 (미지정시 매번 랜덤)")
    args = ap.parse_args()

    # 시드 설정: None이면 현재 시간 기반 랜덤
    if args.seed is None:
        seed = int(time.time() * 1000) % (2**31)
    else:
        seed = args.seed
    
    random.seed(seed)
    torch.manual_seed(seed)

    gender_name = "남성" if args.gender == "M" else "여성"
    print(f"[INFO] {gender_name} 화자 {args.num_speakers}명 혼합 테스트")
    print(f"[INFO] 랜덤 시드: {seed}")
    print("=" * 50)

    # 1) SPEAKERS.TXT 파싱
    print("[STEP 1] SPEAKERS.TXT 파싱 중...")
    gender_map = parse_speakers_txt(SPEAKERS_TXT, subset="test-clean")
    print(f"  - test-clean 남성 화자: {len(gender_map['M'])}명")
    print(f"  - test-clean 여성 화자: {len(gender_map['F'])}명")

    # 2) 데이터셋 로드 및 화자 인덱스 빌드
    print("[STEP 2] LibriSpeech 데이터셋 로딩...")
    ds = load_ds()
    speakers = build_speaker_index(ds)
    print(f"  - 데이터셋 내 화자 수: {len(speakers)}명")

    # 3) 동성 화자 선택
    print(f"[STEP 3] {gender_name} 화자 {args.num_speakers}명 선택...")
    selected_speakers = pick_speakers_by_gender(
        speakers, gender_map[args.gender], args.num_speakers
    )
    tgt_spk = selected_speakers[0]
    interferers = selected_speakers[1:]
    print(f"  - 타겟 화자: {tgt_spk}")
    print(f"  - 방해 화자: {interferers}")

    # 4) Enroll 음성 생성 (타겟 화자 20초)
    print("[STEP 4] Enroll 음성 생성 (20초)...")
    tgt_idxs = random.sample(speakers[tgt_spk], k=min(5, len(speakers[tgt_spk])))
    enroll_long = build_long_audio(ds, tgt_idxs, min_sec=20.0)
    enroll = cut_or_pad(enroll_long, ENROLL_SEC)
    sf.write(OUT_ENROLL, enroll.numpy().astype(np.float32), SR)
    print(f"  - 저장: {OUT_ENROLL}")

    # 5) 각 화자별 10초 오디오 생성
    print("[STEP 5] 각 화자별 10초 오디오 생성...")
    
    def long_audio_for(spk):
        idxs = random.sample(speakers[spk], k=min(6, len(speakers[spk])))
        long = build_long_audio(ds, idxs, min_sec=12.0)
        return random_crop_10sec(long)

    target = long_audio_for(tgt_spk)
    interfs_np = [long_audio_for(spk) for spk in interferers]

    # 6) Ground Truth 저장 (검증용)
    print("[STEP 6] Ground Truth 파일 저장 (검증용)...")
    sf.write(OUT_GT_TARGET, target, SR)
    print(f"  - 저장: {OUT_GT_TARGET} (타겟 화자 원본)")
    
    for i, (spk, interf) in enumerate(zip(interferers, interfs_np)):
        gt_interf_path = f"gt_interf{i}.wav"
        sf.write(gt_interf_path, interf, SR)
        print(f"  - 저장: {gt_interf_path} (방해 화자 {spk} 원본)")

    # 7) SNR 스케일링 및 혼합
    print("[STEP 7] SNR 스케일링 및 혼합...")
    snrs = default_snr_list(len(interfs_np))
    print(f"  - SNR 설정: {snrs}")
    
    sigs = scale_to_snr(target, interfs_np, snrs)
    mix = np.sum(np.stack(sigs, axis=0), axis=0)
    mix = np.clip(mix, -1.0, 1.0).astype(np.float32)

    sf.write(OUT_MIX, mix, SR)
    print(f"  - 저장: {OUT_MIX}")

    print("=" * 50)
    print(f"[완료] {gender_name} 화자 {args.num_speakers}명 혼합 완료!")
    print()
    print("📁 생성된 파일:")
    print(f"  - {OUT_MIX}: 혼합 음성")
    print(f"  - {OUT_ENROLL}: 타겟 화자 등록 음성 (20초)")
    print(f"  - {OUT_GT_TARGET}: 타겟 화자 원본 (10초, 검증용)")
    for i in range(len(interferers)):
        print(f"  - gt_interf{i}.wav: 방해 화자 {interferers[i]} 원본 (검증용)")
    print()
    print("💡 검증 방법: gt_target.wav와 sep_src*.wav를 비교 청취하여")
    print("   어떤 분리 결과가 타겟인지 확인하세요.")
    print()
    print("다음 단계: python 2_sepformer_extract_all.py 실행")


if __name__ == "__main__":
    main()

