# -*- coding: utf-8 -*-
"""
통합 실행 스크립트: Target Speaker Extraction Pipeline

3가지 모드 지원:
1. default: LibriSpeech에서 랜덤 화자 선택 → 혼합 → 분리
2. gender:  LibriSpeech에서 특정 성별 화자만 선택 → 혼합 → 분리
3. custom:  사용자 지정 파일 사용 → 분리

사용 예시:
    python run_all.py --mode default --num_speakers 3
    python run_all.py --mode gender --gender M --num_speakers 3
    python run_all.py --mode custom --mixture "혼합.m4a" --enroll "타겟.m4a"
"""

import os
import sys
import glob
import subprocess
import argparse
from pathlib import Path

os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PY = sys.executable


# ============================================================
# 유틸리티
# ============================================================
def run(cmd, required=True):
    """명령어 실행"""
    print("\n" + "=" * 50)
    print("$ " + " ".join(cmd))
    print("=" * 50)
    try:
        subprocess.check_call(cmd)
        return True
    except subprocess.CalledProcessError as e:
        if required:
            print(f"[ERROR] 실행 실패: {e}")
            raise
        else:
            print(f"[WARN] 실행 실패 (선택 단계): {e}")
            return False


def clean_outputs():
    """이전 결과물 제거"""
    patterns = [
        "sep_src*.wav",
        "tas_src*.wav",
        "target*.wav",
        "gt_*.wav",
        "*.npy"
    ]
    print("\n[CLEAN] 이전 결과물 제거 중...")
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                print(f"  - {f} 삭제")
            except:
                pass


# ============================================================
# 데이터 준비 단계
# ============================================================
def prepare_data_default(num_speakers):
    """기본 모드: LibriSpeech에서 랜덤 화자 선택"""
    print("\n[MODE: DEFAULT] 랜덤 화자 혼합 데이터 생성")
    run([PY, "make_data_from_librispeech_nmix.py", 
         "--num_speakers", str(num_speakers)])


def prepare_data_gender(gender, num_speakers):
    """성별 모드: 특정 성별 화자만 선택"""
    print(f"\n[MODE: GENDER] {gender} 화자 {num_speakers}명 혼합 데이터 생성")
    run([PY, "test_same_gender_mix.py", 
         "--gender", gender, 
         "--num_speakers", str(num_speakers)])


def prepare_data_custom(mixture_path, enroll_path):
    """커스텀 모드: 사용자 지정 파일 전처리"""
    print(f"\n[MODE: CUSTOM] 커스텀 파일 전처리")
    print(f"  - 혼합 음성: {mixture_path}")
    print(f"  - 등록 음성: {enroll_path}")
    
    # test_custom_audio.py의 전처리 부분만 실행
    run([PY, "test_custom_audio.py", 
         "--mixture", mixture_path, 
         "--enroll", enroll_path])


# ============================================================
# 분리 파이프라인
# ============================================================
def run_separation_pipeline():
    """분리 및 후처리 파이프라인 실행"""
    
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
        ("7. 타겟 음성 활성화", "6a_target_voice_activation.py", False),
        ("8. 스펙트럴 ECAPA 마스크", "6b_spectral_ecapa_mask.py", False),
        
        # 최종 후처리
        ("9. Hybrid 후처리", "5_postproc_spectral_sub.py", True),
        ("10. 최종 클린업", "7_final_cleanup.py", False),
    ]
    
    print("\n" + "=" * 60)
    print("🎯 분리 파이프라인 시작")
    print("=" * 60)
    
    for step_name, script, required in steps:
        if not Path(script).exists():
            if required:
                print(f"\n[ERROR] 필수 파일 없음: {script}")
            else:
                print(f"\n[SKIP] 파일 없음: {script}")
            continue
        
        print(f"\n[STEP] {step_name}")
        run([PY, script], required=required)
    
    print("\n" + "=" * 60)
    print("✅ 파이프라인 완료!")
    print("=" * 60)


# ============================================================
# 결과 출력
# ============================================================
def print_results():
    """생성된 결과 파일 출력"""
    print("\n📁 생성된 결과 파일:")
    
    result_files = [
        ("target_emphasized.wav", "기본 분리 결과"),
        ("target_emphasized_maskref.wav", "마스크 리파인"),
        ("target_emphasized_enh.wav", "유사도 강화"),
        ("target_supermask.wav", "Supermask"),
        ("target_superboost.wav", "부스트"),
        ("target_boosted_final.wav", "비타겟 억제"),
        ("target_spectral_boosted.wav", "스펙트럴 마스크"),
        ("target_emphasized_post.wav", "Hybrid 후처리"),
        ("target_final_clean.wav", "최종 클린업 ⭐"),
    ]
    
    for filename, desc in result_files:
        if Path(filename).exists():
            print(f"  ✓ {filename:35s} ({desc})")
    
    print("\n💡 추천: target_final_clean.wav를 먼저 청취해보세요!")


# ============================================================
# 메인
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Target Speaker Extraction - 통합 실행 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 모드: 랜덤 화자 3명 혼합
  python run_all.py --mode default --num_speakers 3

  # 성별 모드: 남성 화자 3명 혼합
  python run_all.py --mode gender --gender M --num_speakers 3

  # 성별 모드: 여성 화자 3명 혼합
  python run_all.py --mode gender --gender F --num_speakers 3

  # 커스텀 모드: 직접 녹음한 파일 사용
  python run_all.py --mode custom --mixture "혼합.m4a" --enroll "타겟.m4a"
        """
    )
    
    # 모드 선택
    ap.add_argument("--mode", type=str, default="default",
                    choices=["default", "gender", "custom"],
                    help="실행 모드 선택")
    
    # default/gender 모드 옵션
    ap.add_argument("--num_speakers", type=int, default=3,
                    help="혼합할 화자 수 (default/gender 모드)")
    
    # gender 모드 옵션
    ap.add_argument("--gender", type=str, default="M",
                    choices=["M", "F"],
                    help="화자 성별 (gender 모드): M(남성) 또는 F(여성)")
    
    # custom 모드 옵션
    ap.add_argument("--mixture", type=str, default=None,
                    help="혼합 음성 파일 경로 (custom 모드)")
    ap.add_argument("--enroll", type=str, default=None,
                    help="타겟 화자 등록 음성 파일 경로 (custom 모드)")
    
    # 공통 옵션
    ap.add_argument("--no-clean", action="store_true",
                    help="이전 결과물 삭제 건너뛰기")
    ap.add_argument("--skip-data", action="store_true",
                    help="데이터 준비 단계 건너뛰기 (이미 mixture.wav, enroll_target_clean.wav가 있는 경우)")
    
    args = ap.parse_args()
    
    # 모드별 검증
    if args.mode == "custom":
        if not args.mixture or not args.enroll:
            print("[ERROR] custom 모드에서는 --mixture와 --enroll이 필수입니다.")
            print("예시: python run_all.py --mode custom --mixture \"혼합.m4a\" --enroll \"타겟.m4a\"")
            sys.exit(1)
        if not Path(args.mixture).exists():
            print(f"[ERROR] 혼합 음성 파일을 찾을 수 없습니다: {args.mixture}")
            sys.exit(1)
        if not Path(args.enroll).exists():
            print(f"[ERROR] 등록 음성 파일을 찾을 수 없습니다: {args.enroll}")
            sys.exit(1)
    
    # 헤더 출력
    print("\n" + "=" * 60)
    print("🎤 Target Speaker Extraction Pipeline")
    print("=" * 60)
    
    mode_names = {
        "default": "기본 (랜덤 화자)",
        "gender": f"성별 지정 ({args.gender})",
        "custom": "커스텀 파일"
    }
    print(f"모드: {mode_names[args.mode]}")
    
    if args.mode in ["default", "gender"]:
        print(f"화자 수: {args.num_speakers}명")
    if args.mode == "custom":
        print(f"혼합 음성: {args.mixture}")
        print(f"등록 음성: {args.enroll}")
    
    # 0. 이전 결과 제거
    if not args.no_clean:
        clean_outputs()
    
    # 1. 데이터 준비
    if not args.skip_data:
        if args.mode == "default":
            prepare_data_default(args.num_speakers)
        elif args.mode == "gender":
            prepare_data_gender(args.gender, args.num_speakers)
        elif args.mode == "custom":
            prepare_data_custom(args.mixture, args.enroll)
    else:
        print("\n[SKIP] 데이터 준비 단계 건너뜀")
    
    # 2. 분리 파이프라인 실행
    run_separation_pipeline()
    
    # 3. 결과 출력
    print_results()


if __name__ == "__main__":
    main()
