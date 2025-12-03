# -*- coding: utf-8 -*-
import os, sys, glob, subprocess
import argparse

os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PY = sys.executable


def run(cmd):
    print("\n$ " + " ".join(cmd))
    subprocess.check_call(cmd)


def clean_outputs():
    """이전 결과물 제거"""
    patterns = [
        "sep_src*.wav",
        "tas_src*.wav",
        "target*.wav",
        "*.npy"
    ]
    for pat in patterns:
        for f in glob.glob(pat):
            try:
                os.remove(f)
                print("[CLEAN] removed", f)
            except:
                pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--num_speakers", type=int, default=6,
                    help="혼합에 사용할 화자 수")
    args = ap.parse_args()

    # -------------------------------------
    # 0. 이전 결과 제거
    # -------------------------------------
    clean_outputs()

    # -------------------------------------
    # 1. N명 혼합 데이터 생성
    # -------------------------------------
    run([
        PY,
        "make_data_from_librispeech_nmix.py",
        "--num_speakers", str(args.num_speakers)
    ])

    # -------------------------------------
    # 2. SepFormer 전체 분리
    # -------------------------------------
    run([PY, "2_sepformer_extract_all.py"])

    # -------------------------------------
    # 3. 타깃 화자 선택
    # -------------------------------------
    run([PY, "4_choose_target_by_embedding_ecapa_vad.py"])

    # -------------------------------------
    # 4. Mask refine (5b)
    # -------------------------------------
    run([PY, "5b_mask_refine.py"])

    # -------------------------------------
    # 5. Similarity Enhance (5c)
    # -------------------------------------
    try:
        run([PY, "5c_similarity_enhance.py"])
    except Exception as e:
        print("[WARN] 5c failed:", e)

    # -------------------------------------
    # 6. Supermask refine (5d)
    # -------------------------------------
    try:
        run([PY, "5d_supermask_refine.py"])
    except Exception as e:
        print("[WARN] 5d failed:", e)

    # -------------------------------------
    # 7. Superboost (5e)
    # -------------------------------------
    try:
        run([PY, "5e_superboost.py"])
    except Exception as e:
        print("[WARN] 5e failed:", e)

    # -------------------------------------
    # 8. Voice Activation (6a)
    # -------------------------------------
    try:
        run([PY, "6a_target_voice_activation.py"])
    except Exception as e:
        print("[WARN] 6a failed:", e)

    # -------------------------------------
    # 9. Spectral ECAPA Mask (6b)
    # -------------------------------------
    try:
        run([PY, "6b_spectral_ecapa_mask.py"])
    except Exception as e:
        print("[WARN] 6b failed:", e)

    # -------------------------------------
    # 10. 최종 후처리 (PostFilter)
    # -------------------------------------
    run([PY, "5_postproc_spectral_sub.py"])
    # -------------------------------------
    # FINISH
    # -------------------------------------
    print("\n[OK] All steps completed!")
    print("Generated outputs:")
    print(" - sep_src*.wav")
    print(" - target_emphasized.wav")
    print(" - target_emphasized_maskref.wav")
    print(" - target_similarity_enh.wav")
    print(" - target_supermask.wav")
    print(" - target_superboost.wav")
    print(" - target_boosted_final.wav")
    print(" - target_spectral_boosted.wav")
    print(" - target_emphasized_post.wav  (최종)")
