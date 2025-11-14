# -*- coding: utf-8 -*-
import os, sys, glob, subprocess
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

PY = sys.executable

def run(cmd):
    print("\n$", " ".join(cmd))
    subprocess.check_call(cmd)

def clean_outputs():
    for pat in ["sep_src*.wav", "tas_src*.wav", "target_emphasized*.wav"]:
        for f in glob.glob(pat):
            try: os.remove(f)
            except: pass

if __name__ == "__main__":
    clean_outputs()
    # ✅ 화자 5명 혼합 생성
    run([PY, "make_data_from_librispeech_nmix.py", "--num_speakers", "5"])

    # 반복 SepFormer로 잔차 분리
    try:
        run([PY, "2_sepformer_iterative.py"])
    except Exception as e:
        print("[WARN] Iterative SepFormer failed:", e)

    run([PY, "4_choose_target_by_embedding_ecapa_vad.py"])
    try:
        run([PY, "5b_mask_refine.py"])
    except Exception as e:
        print("[WARN] Mask refine failed:", e)
    run([PY, "5_postproc_spectral_sub.py"])

    print("\n[OK] Done. Check outputs:")
    print(" - sep_src*.wav (가변 개수; 5명 이상이면 더 많이 생성 가능)")
    print(" - target_emphasized*.wav")
