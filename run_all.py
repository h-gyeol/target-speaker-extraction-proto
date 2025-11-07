import os, subprocess, sys

PY = sys.executable  # 현재 인터프리터(venv) 사용

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

if __name__ == "__main__":
    # 1) 데이터 생성
    run([PY, "make_data_from_librispeech.py"])

    # 2) 분리 (SepFormer → Asteroid 순서; 실패해도 다음 진행)
    try:
        run([PY, "2_sepformer_demo.py"])  # sep_src*.wav
    except Exception as e:
        print("[WARN] SepFormer failed:", e)
        print("      (관리자권한 실행 또는 Windows 개발자 모드 활성화가 필요할 수 있습니다.)")
    try:
        run([PY, "3_asteroid_demo.py"])  # tas_src*.wav
    except Exception as e:
        print("[WARN] Asteroid failed:", e)

    # 3) 임베딩 기반 타깃 선택 (ECAPA + VAD + framewise vote)
    run([PY, "4_choose_target_by_embedding_ecapa_vad.py"])  # target_emphasized.wav

    # 4) 마스크 재합성(지직거림 완화)
    try:
        run([PY, "5b_mask_refine.py"])  # target_emphasized_maskref.wav
    except Exception as e:
        print("[WARN] Mask refine failed:", e)

    # 5) 후처리(스펙트럼 서브트랙션)
    run([PY, "5_postproc_spectral_sub.py"])  # target_emphasized_post.wav

    print("\n[OK] Done. Check outputs:")
    print(" - enroll_target_clean.wav, mixture.wav")
    print(" - sep_src*.wav (if SepFormer succeeded), tas_src*.wav")
    print(" - target_emphasized.wav")
    print(" - target_emphasized_maskref.wav (if refine succeeded)")
    print(" - target_emphasized_post.wav")