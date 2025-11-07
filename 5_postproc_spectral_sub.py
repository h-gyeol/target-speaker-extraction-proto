import numpy as np, soundfile as sf, librosa

def spectral_sub(x, sr=16000, n_fft=512, hop=128, beta=1.1, floor=0.01):
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)
    N = np.percentile(mag, 10, axis=1, keepdims=True)  # 간단 잡음 추정
    est = np.maximum(mag - beta * N, floor)
    Y = est * np.exp(1j*pha)
    y = librosa.istft(Y, hop_length=hop, window='hann', length=len(x))
    # peak normalize
    peak = np.max(np.abs(y));
    if peak > 1: y = y / peak
    return y

if __name__ == "__main__":
    y, sr = sf.read("target_emphasized.wav")
    y2 = spectral_sub(y, sr)
    sf.write("target_emphasized_post.wav", y2, sr)
    print("Saved target_emphasized_post.wav")