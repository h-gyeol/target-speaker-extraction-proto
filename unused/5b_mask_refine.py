import numpy as np
import soundfile as sf
import librosa

def ratio_mask_refine(mix, tgt, sr=16000, n_fft=1024, hop=256, floor=1e-3, p=1.2):
    X = librosa.stft(mix, n_fft=n_fft, hop_length=hop, window='hann')
    T = librosa.stft(tgt, n_fft=n_fft, hop_length=hop, window='hann')
    magX, phaX = np.abs(X), np.angle(X)
    magT = np.abs(T)
    magR = np.maximum(magX - magT, 0.0)
    M = (magT**p) / (magT**p + np.maximum(magR, floor)**p + 1e-8)
    Y = M * X
    y = librosa.istft(Y, hop_length=hop, window='hann', length=len(mix))
    pk = np.max(np.abs(y))
    if pk > 1:
        y = y / pk
    return y

if __name__ == "__main__":
    # Load mixture (likely 16k)
    mix, sr_mix = sf.read("mixture.wav")
    if mix.ndim > 1:
        mix = mix.mean(axis=1)

    # Load target emphasized (likely 8k)
    tgt, sr_tgt = sf.read("target_emphasized.wav")
    if tgt.ndim > 1:
        tgt = tgt.mean(axis=1)

    # === SR AUTO MATCH ===
    # Mask refine requires same sample rate.
    if sr_mix != sr_tgt:
        print(f"[INFO] Resampling mixture from {sr_mix} Hz → {sr_tgt} Hz")
        mix = librosa.resample(mix.astype(float), orig_sr=sr_mix, target_sr=sr_tgt)
        sr = sr_tgt
    else:
        sr = sr_mix

    # Shorten to equal length
    L = min(len(mix), len(tgt))
    mix = mix[:L]
    tgt = tgt[:L]

    print("[INFO] Running Ratio Mask Refinement...")
    y = ratio_mask_refine(mix, tgt, sr=sr)

    sf.write("target_emphasized_maskref.wav", y, sr)
    print("[OK] Saved target_emphasized_maskref.wav (sr=%d)" % sr)
