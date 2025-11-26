import soundfile as sf
from speechbrain.pretrained import SepformerSeparation as Separator

in_wav = "mixture.wav"
separator = Separator.from_hparams(source="speechbrain/sepformer-wsj02mix", savedir="pretrained_sepformer")
ests = separator.separate_file(file=in_wav).cpu().numpy()  # [N, T]

# 주의: ckpt에 따라 8 kHz/16 kHz가 다를 수 있음. 기본 8k로 저장 후 들어보고, 필요하면 16k로 변경.
sr_out = 8000
sf.write("sep_src0.wav", nests[0], sr_out)
sf.write("sep_src1.wav", nests[1], sr_out)
print("Saved sep_src0.wav, sep_src1.wav")