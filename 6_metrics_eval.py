import numpy as np, soundfile as sf
from asteroid.metrics import get_metrics
try:
    from pesq import pesq
    HAS_PESQ = True
except Exception:
    HAS_PESQ = False
from resemblyzer import VoiceEncoder, preprocess_wav

# ref: 타깃 원신호(데이터 생성 시 tar10 같은 것) / est: 결과물

def eval_all(ref, est, sr=16000, enroll_ref_for_sim=None):
    m = get_metrics(ref=ref[None,:], est=est[None,:], sample_rate=sr,
                    metrics_list=["si_sdr","sdr","sir","sar"])
    pesq_wb = None
    if HAS_PESQ and sr == 16000:
        try:
            pesq_wb = pesq(sr, ref, est, 'wb')
        except Exception:
            pesq_wb = None
    sim = None
    if enroll_ref_for_sim is not None:
        enc = VoiceEncoder()
        e1 = enc.embed_utterance(preprocess_wav(est, sr))
        e2 = enc.embed_utterance(preprocess_wav(enroll_ref_for_sim, sr))
        sim = float(np.dot(e1, e2) / (np.linalg.norm(e1)*np.linalg.norm(e2)+1e-8))
    return {**m, "pesq_wb": pesq_wb, "spk_sim": sim}

if __name__ == "__main__":
    # 예시: enroll과 결과물을 불러 평가(참조 신호가 없다면 PESQ/SDR류는 생략)
    ref_path = None  # tar10 원본을 저장했다면 경로 지정
    est_path = "target_emphasized_post.wav"
    est, sr = sf.read(est_path)
    ref = None if ref_path is None else sf.read(ref_path)[0]
    res = eval_all(ref if ref is not None else est, est, sr, enroll_ref_for_sim=est)
    print(res)