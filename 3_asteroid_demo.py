import torch, torchaudio, soundfile as sf
from asteroid.models import ConvTasNet

in_wav = "mixture.wav"
wav, sr = torchaudio.load(in_wav)
if wav.shape[0] > 1:
    wav = wav.mean(0, keepdim=True)
if sr != 16000:
    wav = torchaudio.functional.resample(wav, sr, 16000)
    sr = 16000

model = ConvTasNet.from_pretrained("JorisCos/ConvTasNet_Libri2Mix_sepclean_16k")
model.eval()
with torch.no_grad():
    ests = model(wav)  # [1, N, T] 또는 [N, T]
if ests.dim() == 3:
    ests = ests.squeeze(0)  # -> [N, T]
ests = ests.cpu().numpy()

for i, s in enumerate(ests):
    sf.write(f"tas_src{i}.wav", s, sr)
print("Saved:", [f"tas_src{i}.wav" for i in range(len(ests))])