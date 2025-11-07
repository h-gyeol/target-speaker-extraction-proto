import numpy as np, soundfile as sf, librosa

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
    if pk > 1: y = y / pk
    return y

if __name__ == "__main__":
    mix, sr = sf.read("mixture.wav")
    tgt, sr2 = sf.read("target_emphasized.wav")
    if sr2 != sr:
        raise RuntimeError("SR mismatch; make sure both are same SR")
    y = ratio_mask_refine(mix, tgt, sr=sr)
    sf.write("target_emphasized_maskref.wav", y, sr)
    print("Saved target_emphasized_maskref.wav")