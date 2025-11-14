# -*- coding: utf-8 -*-
import numpy as np, soundfile as sf, sys, glob
from pathlib import Path
from resemblyzer import VoiceEncoder, preprocess_wav

enroll_wav = "enroll_target_clean.wav"
cands = sorted([p for p in glob.glob("sep_src*.wav")] + 
               [p for p in ["tas_src0.wav","tas_src1.wav"] if Path(p).exists()])

if not cands:
    print("[ERROR] 분리 결과 파일이 없습니다.")
    sys.exit(1)

def wav_to_emb(path, enc):
    y, sr = sf.read(path)
    if y.ndim > 1: y = y.mean(axis=1)
    return enc.embed_utterance(preprocess_wav(y, sr))

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

enc = VoiceEncoder()
e_tgt = wav_to_emb(enroll_wav, enc)

best_sim, best_path = -1.0, None
for p in cands:
    s = cos(e_tgt, wav_to_emb(p, enc))
    print(f"{p}: similarity={s:.3f}")
    if s > best_sim:
        best_sim, best_path = s, p

y, sr = sf.read(best_path)
if y.ndim > 1: y = y.mean(axis=1)
sf.write("target_emphasized.wav", y, sr)
print(f"Selected: {best_path} (sim={best_sim:.3f}) -> target_emphasized.wav")
