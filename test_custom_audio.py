# -*- coding: utf-8 -*-
"""
커스텀 오디오 테스트용 전처리 스크립트
- 직접 녹음한 파일을 16kHz mono WAV로 변환
- 다양한 형식 지원 (.m4a, .mp3, .flac, .ogg, .wav 등)
- 변환 후 기존 파이프라인 실행 가능

사용법:
    python test_custom_audio.py --mixture "녹음_혼합.m4a" --enroll "녹음_타겟.m4a"
    python test_custom_audio.py --mixture "녹음_혼합.m4a" --enroll "녹음_타겟.m4a" --run
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import librosa

# pydub은 ffmpeg 필요 (다양한 포맷 지원)
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

# ============================================================
# 설정
# ============================================================
TARGET_SR = 16000
OUT_MIXTURE = "mixture.wav"
OUT_ENROLL = "enroll_target_clean.wav"

SUPPORTED_FORMATS_NATIVE = {".wav", ".flac", ".ogg"}  # soundfile로 직접 읽기 가능
SUPPORTED_FORMATS_PYDUB = {".m4a", ".mp3", ".aac", ".wma", ".aiff"}  # pydub 필요


# ============================================================
# 오디오 로딩 (다양한 형식 지원)
# ============================================================
def load_audio(path: str) -> tuple:
    """
    다양한 형식의 오디오 파일을 로드하여 numpy array로 반환
    Returns: (wav_array, sample_rate)
    """
    path = Path(path)
    ext = path.suffix.lower()
    
    if not path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    
    # 1) soundfile로 직접 읽기 가능한 형식
    if ext in SUPPORTED_FORMATS_NATIVE:
        wav, sr = sf.read(str(path))
        return wav, sr
    
    # 2) pydub이 필요한 형식 (.m4a, .mp3 등)
    if ext in SUPPORTED_FORMATS_PYDUB:
        if not PYDUB_AVAILABLE:
            raise ImportError(
                f"{ext} 형식을 읽으려면 pydub이 필요합니다.\n"
                "설치: pip install pydub\n"
                "또한 ffmpeg이 시스템에 설치되어 있어야 합니다."
            )
        
        print(f"[INFO] pydub으로 {ext} 파일 변환 중...")
        audio = AudioSegment.from_file(str(path))
        
        # numpy array로 변환
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        
        # 스테레오면 reshape
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        
        # 정규화 (-1 ~ 1 범위)
        max_val = 2 ** (audio.sample_width * 8 - 1)
        samples = samples / max_val
        
        return samples, audio.frame_rate
    
    # 3) 그 외: librosa로 시도 (fallback)
    try:
        wav, sr = librosa.load(str(path), sr=None, mono=False)
        if wav.ndim == 2:
            wav = wav.T  # (channels, samples) -> (samples, channels)
        return wav, sr
    except Exception as e:
        raise ValueError(f"지원하지 않는 형식이거나 읽기 실패: {path}\n에러: {e}")


# ============================================================
# 전처리: 16kHz mono WAV로 변환
# ============================================================
def preprocess_audio(wav: np.ndarray, sr: int, target_sr: int = TARGET_SR) -> np.ndarray:
    """
    오디오를 16kHz mono로 변환
    """
    # 1) Mono 변환
    if wav.ndim > 1:
        if wav.shape[1] == 2:  # (samples, channels)
            wav = wav.mean(axis=1)
        elif wav.shape[0] == 2:  # (channels, samples)
            wav = wav.mean(axis=0)
    
    wav = wav.astype(np.float32)
    
    # 2) 리샘플링
    if sr != target_sr:
        print(f"[INFO] 리샘플링: {sr} Hz → {target_sr} Hz")
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    
    # 3) 정규화 (클리핑 방지)
    peak = np.max(np.abs(wav))
    if peak > 1.0:
        wav = wav / peak
    
    return wav


# ============================================================
# 파일 변환 및 저장
# ============================================================
def convert_and_save(input_path: str, output_path: str, description: str):
    """
    입력 파일을 16kHz mono WAV로 변환하여 저장
    """
    print(f"\n[처리 중] {description}")
    print(f"  입력: {input_path}")
    
    # 로드
    wav, sr = load_audio(input_path)
    print(f"  원본: {sr} Hz, shape={wav.shape}")
    
    # 전처리
    wav = preprocess_audio(wav, sr, TARGET_SR)
    print(f"  변환: {TARGET_SR} Hz, mono, {len(wav)} samples ({len(wav)/TARGET_SR:.2f}초)")
    
    # 저장
    sf.write(output_path, wav, TARGET_SR)
    print(f"  저장: {output_path}")


# ============================================================
# 파이프라인 실행
# ============================================================
def run_pipeline():
    """전체 파이프라인 실행 (모든 후처리 단계 포함)"""
    PY = sys.executable
    
    # 전체 파이프라인 단계 (run_all.py 참고)
    steps = [
        # 기본 분리
        ("1. SepFormer 분리", "2_sepformer_extract_all.py", True),
        ("2. 타겟 화자 선택", "4_choose_target_by_embedding_ecapa_vad.py", True),
        
        # 마스크 기반 후처리
        ("3. 마스크 리파인", "5b_mask_refine.py", True),
        ("4. 유사도 기반 강화", "5c_similarity_enhance.py", False),
        ("5. Supermask 리파인", "5d_supermask_refine.py", False),
        ("6. Superboost", "5e_superboost.py", False),
        
        # 추가 강화
        ("7. 타겟 음성 활성화 (비타겟 억제)", "6a_target_voice_activation.py", False),
        ("8. 스펙트럴 ECAPA 마스크", "6b_spectral_ecapa_mask.py", False),
        
        # 최종 후처리
        ("9. 최종 후처리 (Hybrid PostFilter)", "5_postproc_spectral_sub.py", True),
        
        # 최종 클린업 (잔여 노이즈/방해 화자 제거)
        ("10. 최종 클린업", "7_final_cleanup.py", False),
    ]
    
    print("\n" + "=" * 50)
    print("[전체 파이프라인 실행]")
    print("=" * 50)
    
    for name, script, required in steps:
        if not Path(script).exists():
            if required:
                print(f"[ERROR] 필수 파일 {script}이 없습니다!")
            else:
                print(f"[SKIP] {script} 파일이 없습니다.")
            continue
        
        print(f"\n[STEP] {name}...")
        try:
            subprocess.check_call([PY, script])
        except subprocess.CalledProcessError as e:
            if required:
                print(f"[ERROR] {script} 실행 실패: {e}")
            else:
                print(f"[WARN] {script} 실행 실패 (선택 단계): {e}")
    
    print("\n" + "=" * 50)
    print("[완료] 전체 파이프라인 실행 완료!")
    print("=" * 50)
    print("\n📁 생성된 결과 파일:")
    print("  - target_emphasized.wav        (기본 분리 결과)")
    print("  - target_emphasized_maskref.wav (마스크 리파인)")
    print("  - target_emphasized_enh.wav    (유사도 강화)")
    print("  - target_supermask.wav         (Supermask)")
    print("  - target_superboost.wav        (부스트)")
    print("  - target_boosted_final.wav     (비타겟 억제)")
    print("  - target_emphasized_post.wav   (Hybrid 후처리)")
    print("  - target_final_clean.wav       (최종 클린업) ⭐")
    print("\n💡 추천: target_final_clean.wav를 먼저 청취해보세요!")
    print("=" * 50)


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="커스텀 오디오 전처리 및 테스트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예시:
  python test_custom_audio.py --mixture "녹음_혼합.m4a" --enroll "녹음_타겟.m4a"
  python test_custom_audio.py --mixture "녹음_혼합.mp3" --enroll "녹음_타겟.wav" --run

지원 형식:
  .wav, .flac, .ogg (기본)
  .m4a, .mp3, .aac, .wma, .aiff (pydub + ffmpeg 필요)
        """
    )
    ap.add_argument("--mixture", type=str, required=True,
                    help="혼합 음성 파일 경로 (여러 사람이 말하는 녹음)")
    ap.add_argument("--enroll", type=str, required=True,
                    help="타겟 화자 등록 음성 파일 경로 (강조하고 싶은 사람의 단독 녹음)")
    ap.add_argument("--run", action="store_true",
                    help="변환 후 파이프라인 자동 실행")
    
    args = ap.parse_args()
    
    print("=" * 50)
    print("🎤 커스텀 오디오 전처리")
    print("=" * 50)
    
    # 1) Mixture 변환
    convert_and_save(args.mixture, OUT_MIXTURE, "혼합 음성 (mixture)")
    
    # 2) Enroll 변환
    convert_and_save(args.enroll, OUT_ENROLL, "타겟 화자 등록 음성 (enroll)")
    
    print("\n" + "=" * 50)
    print("✅ 전처리 완료!")
    print(f"  - {OUT_MIXTURE}: 혼합 음성 (16kHz mono)")
    print(f"  - {OUT_ENROLL}: 타겟 화자 등록 음성 (16kHz mono)")
    print("=" * 50)
    
    # 3) 파이프라인 실행 (옵션)
    if args.run:
        run_pipeline()
    else:
        print("\n다음 단계:")
        print("  python 2_sepformer_extract_all.py")
        print("  python 4_choose_target_by_embedding_ecapa_vad.py")
        print("  ...")
        print("\n또는 --run 옵션으로 자동 실행:")
        print(f"  python test_custom_audio.py --mixture \"{args.mixture}\" --enroll \"{args.enroll}\" --run")


if __name__ == "__main__":
    main()


