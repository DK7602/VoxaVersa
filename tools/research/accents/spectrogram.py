import sys, numpy as np, soundfile as sf, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
out, t0, t1 = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]); files = sys.argv[4:]
fig, ax = plt.subplots(len(files), 1, figsize=(14, 2.6 * len(files)), sharex=True)
for i, f in enumerate(files):
    x, sr = sf.read(f, dtype="float32", always_2d=True); x = x.mean(1); a, b = int(t0 * sr), int(t1 * sr)
    ax[i].specgram(x[a:b], NFFT=1024, Fs=sr, noverlap=768, cmap="magma", vmin=-130, xextent=(t0, t1)); ax[i].set_ylim(0, 9000)
    ax[i].set_title(f.split("/")[-1], fontsize=9)
    for s in (1.84, 2.06, 2.70, 2.99, 3.26, 3.33, 4.15, 4.31, 5.32, 5.54): ax[i].axvline(s, color="c", lw=0.6)
plt.tight_layout(); plt.savefig(out, dpi=60)
