import torchaudio, torch, soundfile as sf, sys, os

def to_mono16k(in_path, out_path, target_sr=16000):
    wav, sr = torchaudio.load(in_path)             # [C, T]
    if wav.dtype != torch.float32: wav = wav.float()
    if wav.shape[0] > 1: wav = wav.mean(0, keepdim=True)
    if sr != target_sr:
        wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
        sr = target_sr
    peak = wav.abs().max()
    if peak > 1: wav = wav / peak
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    sf.write(out_path, wav.squeeze(0).numpy(), sr)

if __name__ == "__main__":
    to_mono16k(sys.argv[1], sys.argv[2])