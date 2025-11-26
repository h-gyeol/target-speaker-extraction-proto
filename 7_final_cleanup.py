# -*- coding: utf-8 -*-
"""
최종 클린업 단계: 잔여 노이즈 및 방해 화자 음성 제거
- 적응형 노이즈 게이트
- 스펙트럴 감산 강화
- 고주파 노이즈 필터
- 부드러운 음성 향상

입력: target_boosted_final.wav (또는 지정 파일)
출력: target_final_clean.wav
"""

import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import median_filter, uniform_filter1d
import argparse


def adaptive_noise_gate(y, sr, frame_ms=30, hop_ms=15, 
                        threshold_ratio=0.15, attack_ms=5, release_ms=50):
    """
    적응형 노이즈 게이트
    - 저에너지 구간을 부드럽게 억제
    - 급격한 변화 방지를 위한 attack/release 적용
    """
    frame_len = int(sr * frame_ms / 1000)
    hop_len = int(sr * hop_ms / 1000)
    
    # 프레임별 에너지 계산
    energies = []
    for start in range(0, len(y) - frame_len, hop_len):
        frame = y[start:start + frame_len]
        energy = np.sqrt(np.mean(frame ** 2))
        energies.append(energy)
    
    energies = np.array(energies)
    
    # 적응형 임계값 (상위 에너지의 일정 비율)
    threshold = np.percentile(energies, 75) * threshold_ratio
    
    # 게이트 마스크 생성 (0 ~ 1)
    gate = np.clip((energies - threshold * 0.5) / (threshold * 0.5 + 1e-8), 0, 1)
    
    # 스무딩 (attack/release 효과)
    attack_frames = max(1, int(attack_ms / hop_ms))
    release_frames = max(1, int(release_ms / hop_ms))
    
    # 상승 시 빠르게, 하강 시 느리게
    smoothed_gate = np.zeros_like(gate)
    smoothed_gate[0] = gate[0]
    for i in range(1, len(gate)):
        if gate[i] > smoothed_gate[i-1]:
            # Attack (빠르게 열림)
            alpha = 1.0 / attack_frames
        else:
            # Release (천천히 닫힘)
            alpha = 1.0 / release_frames
        smoothed_gate[i] = smoothed_gate[i-1] + alpha * (gate[i] - smoothed_gate[i-1])
    
    # 게이트를 원본 길이로 확장
    gate_full = np.interp(
        np.arange(len(y)),
        np.arange(len(smoothed_gate)) * hop_len + frame_len // 2,
        smoothed_gate
    )
    gate_full = np.clip(gate_full, 0.05, 1.0)  # 완전히 0이 되지 않도록
    
    return y * gate_full, gate_full


def enhanced_spectral_subtraction(y, sr, n_fft=512, hop=128, 
                                   beta=1.5, floor=0.02):
    """
    강화된 스펙트럴 감산
    - 잔여 노이즈 제거에 효과적
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)
    
    # 노이즈 추정 (하위 15% 퍼센타일)
    noise_est = np.percentile(mag, 15, axis=1, keepdims=True)
    
    # 스펙트럴 감산
    mag_clean = np.maximum(mag - beta * noise_est, floor * mag)
    
    # 스펙트럴 스무딩 (musical noise 방지)
    mag_clean = median_filter(mag_clean, size=(3, 1))
    
    # 재합성
    S_clean = mag_clean * np.exp(1j * pha)
    y_clean = librosa.istft(S_clean, hop_length=hop, window='hann', length=len(y))
    
    return y_clean


def highfreq_noise_filter(y, sr, cutoff_hz=7500, rolloff_db=12):
    """
    고주파 노이즈 필터
    - 고주파 치찰음/잡음 완화
    - 부드러운 롤오프로 자연스러움 유지
    """
    # STFT
    n_fft = 512
    hop = 128
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)
    
    # 주파수 빈 계산
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    
    # 롤오프 마스크 생성
    rolloff_mask = np.ones_like(freqs)
    for i, f in enumerate(freqs):
        if f > cutoff_hz:
            # 점진적 롤오프
            db_reduction = rolloff_db * (f - cutoff_hz) / (sr/2 - cutoff_hz)
            rolloff_mask[i] = 10 ** (-db_reduction / 20)
    
    rolloff_mask = rolloff_mask[:, np.newaxis]  # (freq, 1)
    
    # 적용
    mag_filtered = mag * rolloff_mask
    S_filtered = mag_filtered * np.exp(1j * pha)
    y_filtered = librosa.istft(S_filtered, hop_length=hop, window='hann', length=len(y))
    
    return y_filtered


def residual_suppression(y, sr, n_fft=512, hop=128, 
                         suppress_ratio=0.3, smooth_frames=5):
    """
    잔여 신호 억제
    - 낮은 에너지의 잔여 성분 추가 억제
    """
    S = librosa.stft(y, n_fft=n_fft, hop_length=hop, window='hann')
    mag, pha = np.abs(S), np.angle(S)
    
    # 프레임별 에너지
    frame_energy = np.sum(mag ** 2, axis=0)
    
    # 에너지 스무딩
    frame_energy_smooth = uniform_filter1d(frame_energy, size=smooth_frames)
    
    # 적응형 억제 마스크
    energy_norm = frame_energy_smooth / (np.max(frame_energy_smooth) + 1e-8)
    suppress_mask = np.clip(energy_norm / suppress_ratio, 0, 1)
    suppress_mask = suppress_mask ** 0.5  # 부드러운 곡선
    
    # 마스크 적용
    mag_suppressed = mag * suppress_mask[np.newaxis, :]
    
    S_out = mag_suppressed * np.exp(1j * pha)
    y_out = librosa.istft(S_out, hop_length=hop, window='hann', length=len(y))
    
    return y_out


def final_cleanup(input_path, output_path, 
                  gate_threshold=0.15,
                  spectral_beta=1.5,
                  highfreq_cutoff=7500,
                  suppress_ratio=0.3,
                  verbose=True):
    """
    최종 클린업 파이프라인
    """
    if verbose:
        print(f"[INFO] 입력 파일: {input_path}")
    
    # 1) 로드
    y, sr = sf.read(input_path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    y = y.astype(np.float32)
    
    if verbose:
        print(f"[INFO] 샘플레이트: {sr} Hz, 길이: {len(y)/sr:.2f}초")
    
    # 2) 적응형 노이즈 게이트
    if verbose:
        print("[STEP 1] 적응형 노이즈 게이트...")
    y, gate = adaptive_noise_gate(y, sr, threshold_ratio=gate_threshold)
    
    # 3) 강화된 스펙트럴 감산
    if verbose:
        print("[STEP 2] 스펙트럴 감산 강화...")
    y = enhanced_spectral_subtraction(y, sr, beta=spectral_beta)
    
    # 4) 잔여 신호 억제
    if verbose:
        print("[STEP 3] 잔여 신호 억제...")
    y = residual_suppression(y, sr, suppress_ratio=suppress_ratio)
    
    # 5) 고주파 노이즈 필터
    if verbose:
        print("[STEP 4] 고주파 노이즈 필터...")
    y = highfreq_noise_filter(y, sr, cutoff_hz=highfreq_cutoff)
    
    # 6) 최종 정규화
    peak = np.max(np.abs(y))
    if peak > 0.95:
        y = y * 0.95 / peak
    elif peak < 0.5:
        y = y * 0.7 / peak  # 너무 작으면 살짝 키움
    
    # 7) 저장
    sf.write(output_path, y, sr)
    
    if verbose:
        print(f"[OK] 저장 완료: {output_path}")
    
    return y, sr


def main():
    ap = argparse.ArgumentParser(description="최종 클린업: 잔여 노이즈 및 방해 화자 제거")
    ap.add_argument("--input", type=str, default="target_emphasized_post.wav",
                    help="입력 파일 (기본: target_emphasized_post.wav)")
    ap.add_argument("--output", type=str, default="target_final_clean.wav",
                    help="출력 파일 (기본: target_final_clean.wav)")
    ap.add_argument("--gate", type=float, default=0.05,
                    help="노이즈 게이트 임계값 (0.05~0.3, 높을수록 강함)")
    ap.add_argument("--beta", type=float, default=1.0,
                    help="스펙트럴 감산 강도 (0.8~2.5, 높을수록 강함)")
    ap.add_argument("--highfreq", type=int, default=7500,
                    help="고주파 컷오프 (Hz)")
    ap.add_argument("--suppress", type=float, default=0.15,
                    help="잔여 억제 비율 (0.1~0.5)")
    
    args = ap.parse_args()
    
    print("=" * 50)
    print("🧹 최종 클린업")
    print("=" * 50)
    
    final_cleanup(
        input_path=args.input,
        output_path=args.output,
        gate_threshold=args.gate,
        spectral_beta=args.beta,
        highfreq_cutoff=args.highfreq,
        suppress_ratio=args.suppress
    )
    
    print("=" * 50)
    print("💡 노이즈가 더 남아있다면:")
    print("   --gate 0.1 --beta 1.2 --suppress 0.2")
    print("💡 더 강하게 제거하려면:")
    print("   --gate 0.15 --beta 1.5 --suppress 0.3")
    print("=" * 50)


if __name__ == "__main__":
    main()

