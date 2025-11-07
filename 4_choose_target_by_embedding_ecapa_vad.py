import numpy as np, soundfile as sf, sys
from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path

enroll_wav = "enroll_target_clean.wav"
cands = [p for p in ["sep_src0.wav","sep_src1.wav","tas_src0.wav","tas_src1.wav"] if Path(p).exists()]

if not cands:
    print("[ERROR] 분리 결과 파일이 없습니다. 먼저 2_sepformer_demo.py 또는 3_asteroid_demo.py를 성공시켜 주세요.")
    sys.exit(1)

def wav_to_emb(path, enc):
    y, sr = sf.read(path)
    return enc.embed_utterance(preprocess_wav(y, sr))

def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-8))

enc = VoiceEncoder()
e_tgt = wav_to_emb(enroll_wav, enc)

best_sim, best_path = -1.0, None
for p in cands:
    e = wav_to_emb(p, enc)
    s = cos(e_tgt, e)
    print(f"{p}: similarity={s:.3f}")
    if s > best_sim:
        best_sim, best_path = s, p

y, sr = sf.read(best_path)
sf.write("target_emphasized.wav", y, sr)
print(f"Selected: {best_path} (sim={best_sim:.3f}) -> target_emphasized.wav")
