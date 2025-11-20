import numpy as np
import soundfile as sf
from pesq import pesq
from resemblyzer import VoiceEncoder, preprocess_wav
from speechbrain.pretrained import EncoderClassifier
import torch

# ===============================================
# 1) 직접 구현한 SI-SDR / SDR / SIR / SAR
# ===============================================

def sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    return 10 * np.log10(np.sum(ref**2) / np.sum((ref - est)**2) + 1e-10)

def si_sdr(ref, est):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    alpha = np.dot(ref, est) / (np.dot(ref, ref) + 1e-10)
    proj = alpha * ref
    noise = est - proj
    return 10 * np.log10(np.sum(proj**2) / np.sum(noise**2) + 1e-10)

def sir(ref, est, noise):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    noise = noise - np.mean(noise)
    return 10 * np.log10(np.sum(ref**2) / np.sum((est - ref)**2 + np.sum(noise**2)) + 1e-10)

def sar(ref, est, noise):
    ref = ref - np.mean(ref)
    est = est - np.mean(est)
    noise = noise - np.mean(noise)
    e_total = est - ref
    return 10 * np.log10(np.sum(ref**2) / np.sum(e_total**2 + np.sum(noise**2)) + 1e-10)

# ===============================================
# 2) 임베딩 유사도
# ===============================================

def sim_resem(w1, w2):
    enc = VoiceEncoder()
    e1 = enc.embed_utterance(w1.astype(np.float32))
    e2 = enc.embed_utterance(w2.astype(np.float32))
    return float(np.dot(e1,e2)/(np.linalg.norm(e1)*np.linalg.norm(e2)+1e-10))

def sim_ecapa(w1, w2):
    model = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

    # numpy → torch tensor 변환
    t1 = torch.tensor(w1, dtype=torch.float32).unsqueeze(0)  # (1, T)
    t2 = torch.tensor(w2, dtype=torch.float32).unsqueeze(0)

    # ECAPA 임베딩 계산
    e1 = model.encode_batch(t1).squeeze().detach().cpu().numpy()
    e2 = model.encode_batch(t2).squeeze().detach().cpu().numpy()

    # 코사인 유사도
    return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10))

# ===============================================
# 3) 모든 평가 합치기
# ===============================================

def eval_all(ref, est, sr=16000):
    return {
        "si_sdr": si_sdr(ref, est),
        "sdr": sdr(ref, est),
        "sir": None,   # noise 큐가 없으므로 향후 추가
        "sar": None,   # noise 큐가 없으므로 향후 추가
    }

# ===============================================
# 4) 출력
# ===============================================

def print_eval_table(ref_wav, est_wav, sr=16000):
    res = eval_all(ref_wav, est_wav, sr)
    s_resem = sim_resem(ref_wav, est_wav)
    s_ecapa = sim_ecapa(ref_wav, est_wav)

    try:
        p = pesq(sr, ref_wav, est_wav, 'wb')
    except:
        p = None

    print("\n===== Evaluation Summary =====")
    print(f"SI-SDR       : {res['si_sdr']:.4f}")
    print(f"SDR          : {res['sdr']:.4f}")
    print(f"SIR          : N/A")
    print(f"SAR          : N/A")
    print(f"PESQ-WB      : {p if p is not None else 'N/A'}")
    print(f"ReSem-Sim    : {s_resem:.4f}")
    print(f"ECAPA-Sim    : {s_ecapa:.4f}")
    print("==============================\n")

# ===============================================
# 5) 메인
# ===============================================

if __name__ == "__main__":
    # est = 후처리된 타깃 음성
    est, sr = sf.read("target_emphasized_post.wav")
    if est.ndim > 1:
        est = est.mean(axis=1)

    # ref = 깨끗한 타깃 화자 음성
    ref, sr_ref = sf.read("enroll_target_clean.wav")
    if ref.ndim > 1:
        ref = ref.mean(axis=1)

    # 길이 맞추기
    min_len = min(len(ref), len(est))
    ref = ref[:min_len]
    est = est[:min_len]

    # 평가 실행
    print_eval_table(ref, est, sr)