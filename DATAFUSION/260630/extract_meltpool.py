from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from nptdms import TdmsFile

TDMS_PATH = Path("/mnt/a/ftk3187/HAMMER/DED/RGB/raw_test/HAMMER_Square_600W_1_12.tdms")
OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP = "Image Data"
CHANNEL = "Flattened Images"
THRESHOLDS = [0.30e7, 0.35e7, 0.40e7]   # melt pool boundary levels
THRESH_COLORS = ["cyan", "lime", "magenta"]
N_SAMPLES = 100   # frames sampled evenly across the whole dataset

with TdmsFile.open(TDMS_PATH) as tdms:
    ch = tdms[GROUP][CHANNEL]
    w = int(ch.properties["Image X"])
    h = int(ch.properties["Image Y"])
    frame_size = w * h
    n_frames = len(ch) // frame_size

    idxs = np.linspace(0, n_frames - 1, N_SAMPLES, dtype=int)
    print(f"Image {w}x{h}, {n_frames} frames; thresholds={[f'{t:g}' for t in THRESHOLDS]}")
    print(f"Sampling {len(idxs)} frames evenly across whole dataset")

    frames = []
    for fi in idxs:
        data = ch.read_data(offset=int(fi) * frame_size, length=frame_size)
        frames.append(data.reshape(h, w).astype(np.float64))


def area_above(frame, thresh):
    """Pixel count of the largest connected region above `thresh`."""
    binary = frame >= thresh
    if not binary.any():
        return 0
    labels, n = ndimage.label(binary)
    sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
    return int(sizes.max())


ncols = 10
nrows = int(np.ceil(len(idxs) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(2.4 * ncols, 2.7 * nrows))
axes = np.atleast_1d(axes).ravel()

for ax, fi, frame in zip(axes, idxs, frames):
    ax.imshow(frame, cmap="inferno")
    areas = []
    for thr, col in zip(THRESHOLDS, THRESH_COLORS):
        ax.contour(frame, levels=[thr], colors=col, linewidths=1.0)
        areas.append(area_above(frame, thr))
    area_str = "/".join(f"{a/1000:.1f}" for a in areas)
    ax.set_title(f"f{fi}\n{area_str}k", fontsize=6)
    ax.axis("off")
for ax in axes[len(idxs):]:
    ax.axis("off")

# legend mapping color -> threshold
handles = [
    plt.Line2D([0], [0], color=c, lw=2, label=f"{t:g}")
    for t, c in zip(THRESHOLDS, THRESH_COLORS)
]
fig.legend(handles=handles, loc="upper right", ncol=len(THRESHOLDS), fontsize=9)
fig.suptitle(
    f"{TDMS_PATH.name}  melt-pool boundaries (area per threshold, largest blob)  ({w}x{h})",
    fontsize=11,
)
fig.tight_layout()
out = OUT_DIR / f"meltpool_boundary_n{N_SAMPLES}.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")
