import numpy as np, soundfile as sf, librosa
import librosa
import numpy as np
from scipy.ndimage import median_filter

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

def soft_spectral_gate(x, sr=16000, n_fft=512, hop=128, 
                       reduction=0.4, floor=0.02, smooth=3):
    """
    Soft Spectral Gating
    - 목적: 잔여 화자 / 모델 아티팩트 / non-stationary 노이즈 억제
    - spectral subtraction보다 more speech-preserving
    """
    # 1) STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)

    # 2) Noise floor 추정 (프레임별 median 기반 → speech-like noise에 강함)
    noise_est = median_filter(mag, size=(1, smooth))

    # 3) soft mask 생성
    # mask = 1 - reduction*(noise / mag)
    # → 음성이 크게 나오면 mask ~ 1
    # → 잔여 화자 · tail voice는 mask 감소
    eps = 1e-6
    ratio = noise_est / (mag + eps)
    mask = 1.0 - reduction * ratio

    # 4) floor 이하로 떨어지지 않도록
    mask = np.clip(mask, floor, 1.0)

    # 5) smoothing (주파수 영역 median filter)
    mask = median_filter(mask, size=(smooth, 1))

    # 6) apply mask
    Y = mask * mag * np.exp(1j * pha)

    # 7) ISTFT
    y = librosa.istft(Y, hop_length=hop, window='hann', length=len(x))

    # 8) peak normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y = y / peak

    return y

def irm_wiener_refine(x, sr=16000, n_fft=512, hop=128, 
                      alpha=0.7, floor=0.02, smooth=5):
    """
    IRM-like Wiener Mask Refinement
    - 잔여 화자 / tail voice 제거 능력
    - Soft Gate 이후 적용하면 효과 극대화됨
    """
    # 1) STFT
    S = librosa.stft(x, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)

    # 2) Noise 추정 (더 aggressive)
    #    soft gate noise보다 미세 성분도 잡히도록 percentile 25 사용
    noise_est = np.percentile(mag, 25, axis=1, keepdims=True)

    # 3) IRM-like mask 계산
    #    mask = S / (S + alpha*N)
    eps = 1e-6
    mask = mag / (mag + alpha * noise_est + eps)

    # 4) floor 이하로 떨어지지 않도록 clip
    mask = np.clip(mask, floor, 1.0)

    # 5) smoothing (주파수축 median filter)
    mask = median_filter(mask, size=(smooth, 1))

    # 6) apply mask
    Y = mask * mag * np.exp(1j * pha)

    # 7) ISTFT
    y = librosa.istft(Y, hop_length=hop, window='hann', length=len(x))

    # 8) peak normalize
    peak = np.max(np.abs(y))
    if peak > 1:
        y = y / peak

    return y


def apply_postfilter(x, sr=16000, mode="hybrid"):
    """
    Unified advanced post-filter
    mode:
        - "none": 입력 그대로
        - "gate": Soft Spectral Gate 단독
        - "irm": IRM/Wiener refine 단독
        - "hybrid": SoftGate → IRM refine (권장)
    """

    # STFT 파라미터 자동 결정 (8k/16k 호환)
    if sr == 8000:
        n_fft = 256
        hop = 64
    else:
        n_fft = 512
        hop = 128

    if mode == "none":
        return x

    # 1) Soft Gate
    y = soft_spectral_gate(
        x, sr=sr, n_fft=n_fft, hop=hop,
        reduction=0.4, floor=0.02, smooth=3
    )

    if mode == "gate":
        return y

    # 2) IRM refinement
    y2 = irm_wiener_refine(
        y, sr=sr, n_fft=n_fft, hop=hop,
        alpha=0.7, floor=0.02, smooth=5
    )

    # Final peak normalization
    peak = np.max(np.abs(y2))
    if peak > 1:
        y2 = y2 / peak

    return y2

if __name__ == "__main__":
    import sys

    # 입력 파일 결정
    inp = "target_emphasized.wav"
    if len(sys.argv) >= 2:
        inp = sys.argv[1]

    print(f"[INFO] Loading {inp} ...")
    y, sr = sf.read(inp)
    if y.ndim > 1:
        y = y.mean(axis=1)

    # 새로운 Hybrid PostFilter 적용
    print(f"[INFO] Applying Hybrid PostFilter (soft gate + IRM refine)...")
    y2 = apply_postfilter(y, sr, mode="hybrid")

    out_path = "target_emphasized_post.wav"
    sf.write(out_path, y2, sr)

    print(f"[INFO] Saved: {out_path}")