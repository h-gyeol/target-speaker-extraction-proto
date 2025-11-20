# -*- coding: utf-8 -*-
import os
os.environ["SPEECHBRAIN_LOCAL_CACHE_STRATEGY"] = "copy"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import time
import numpy as np
import soundfile as sf
from pathlib import Path
from speechbrain.inference import SepformerSeparation as Separator

MIX_PATH = "mixture.wav"
SAVEDIR  = "pretrained_sepformer_3mix"

def to_sources_2d(x):
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim == 3:
        x = x[0]
    if x.ndim != 2:
        raise ValueError(f"bad shape: {x.shape}")
    N, T = x.shape
    if N > T and T < 10:
        x = x.T
    return x

def main():
    if not Path(MIX_PATH).exists():
        raise FileNotFoundError(MIX_PATH)

    print("[INFO] loading SepFormer...")
    t0 = time.time()
    separator = Separator.from_hparams(source="speechbrain/sepformer-wsj03mix",
                                       savedir=SAVEDIR)
    print(f"[OK] loaded in {time.time()-t0:.2f}s")

    print("[INFO] separating...")
    est = separator.separate_file(MIX_PATH)     # torch
    est = est.detach().cpu().numpy()            # numpy
    est = to_sources_2d(est)                    # [N, T]

    print("[INFO] sources:", est.shape)

    # save all sources
    out_paths = []
    for i in range(est.shape[0]):
        y = est[i].astype(np.float32)
        if y.ndim > 1:
            y = y.mean(axis=-1)

        # peak normalize
        peak = np.max(np.abs(y))
        if peak > 1:
            y = y / peak

        out = f"sep_src{i}.wav"
        sf.write(out, y, 8000)
        out_paths.append(out)

    print("[OK] saved:", out_paths)

if __name__ == "__main__":
    main()
