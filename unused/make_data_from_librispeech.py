import os, random, torch, soundfile as sf
import torchaudio
import numpy as np

TARGET_SR = 16000
ROOT = os.path.abspath("./data_librispeech")
os.makedirs(ROOT, exist_ok=True)

def to_mono16k(wav, sr):
    if wav.ndim == 2:  # [C, T]
        wav = wav.mean(dim=0, keepdim=True)
    if sr != TARGET_SR:
        wav = torchaudio.transforms.Resample(sr, TARGET_SR)(wav)
    wav = wav.squeeze(0)
    peak = wav.abs().max()
    if peak > 1: wav = wav / peak
    return wav

def random_chunk(wav, sec):
    T = int(sec * TARGET_SR)
    if wav.numel() <= T:
        pad = T - wav.numel()
        return torch.nn.functional.pad(wav, (0, pad))
    start = random.randint(0, wav.numel()-T)
    return wav[start:start+T]

def mix_at_snr(target, interferer, snr_db=12.0):  # ← 5.0 → 12.0
    t_p = target.pow(2).mean().item() + 1e-12
    i_p = interferer.pow(2).mean().item() + 1e-12
    scale = np.sqrt(t_p / (i_p * 10**(snr_db/10)))
    mixed = target + torch.tensor(scale, dtype=interferer.dtype) * interferer
    peak = mixed.abs().max().item()
    if peak > 1: mixed = mixed / peak
    return mixed

def main():
    ds = torchaudio.datasets.LIBRISPEECH(ROOT, url="test-clean", download=True)

    by_spk = {}
    for i in range(len(ds)):
        wav, sr, _, spk_id, _, _ = ds[i]
        wav = to_mono16k(wav, sr)
        by_spk.setdefault(spk_id, []).append(wav)

    spks = list(by_spk.keys())
    random.shuffle(spks)
    target_spk, interferer_spk = spks[0], spks[1]

    target_wavs = by_spk[target_spk]
    enroll = torch.cat(target_wavs, dim=0)
    enroll = random_chunk(enroll, sec=20.0)
    sf.write("enroll_target_clean.wav", enroll.numpy(), TARGET_SR)

    tar10 = random_chunk(torch.cat(target_wavs, dim=0), sec=10.0)
    int10 = random_chunk(torch.cat(by_spk[interferer_spk], dim=0), sec=10.0)
    mix = mix_at_snr(tar10, int10, snr_db=12.0)
    sf.write("mixture.wav", mix.numpy(), TARGET_SR)

    print(f"[OK] Target speaker: {target_spk}, Interferer: {interferer_spk}")
    print("Saved: enroll_target_clean.wav (20s), mixture.wav (10s)")

if __name__ == "__main__":
    main()
