from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from nptdms import TdmsFile

TDMS_PATH = Path("/mnt/a/ftk3187/HAMMER/DED/RGB/raw_test/HAMMER_Square_600W_1_12.tdms")
OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP = "Image Data"
CHANNEL = "Flattened Images"
N_SAMPLES = 20   # how many frames to visualize
INTERVAL = 300   # frame step between samples
START = 0        # first frame index

with TdmsFile.open(TDMS_PATH) as tdms:
    ch = tdms[GROUP][CHANNEL]

    # Real dimensions come from the channel properties (Image X / Image Y)
    w = int(ch.properties["Image X"])
    h = int(ch.properties["Image Y"])
    frame_size = w * h
    n_frames = len(ch) // frame_size
    print(f"Image {w}x{h}, {n_frames} frames total, dtype={ch.dtype}")

    # fixed-interval frame indices
    idxs = START + np.arange(N_SAMPLES) * INTERVAL
    idxs = idxs[idxs < n_frames]
    print(f"Sampling {len(idxs)} frames (step {INTERVAL}): {idxs.tolist()}")

    frames = []
    for fi in idxs:
        data = ch.read_data(offset=int(fi) * frame_size, length=frame_size)
        frames.append(data.reshape(h, w))

# montage grid
ncols = 5
nrows = int(np.ceil(len(idxs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.6 * nrows))
axes = np.atleast_1d(axes).ravel()
for ax, fi, frame in zip(axes, idxs, frames):
    im = ax.imshow(frame, cmap="inferno")
    ax.set_title(f"frame {fi}\nmin={frame.min()} max={frame.max()}", fontsize=8)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
for ax in axes[len(idxs):]:
    ax.axis("off")

fig.suptitle(f"{TDMS_PATH.name}  ({w}x{h}, {n_frames} frames)", fontsize=11)
fig.tight_layout()
out = OUT_DIR / "sample_frames.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
