"""One pass over the whole file: per-frame melt-pool centroid / area / shape.
Reads in large contiguous blocks (fast) and caches tracks to npz so all tilt
analysis afterwards is instant (no 42GB re-read).
"""
from pathlib import Path
import numpy as np
from scipy import ndimage
from nptdms import TdmsFile

TDMS_PATH = Path("/mnt/a/ftk3187/HAMMER/DED/RGB/raw_test/HAMMER_Square_600W_1_12.tdms")
OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
GROUP, CHANNEL = "Image Data", "Flattened Images"
THRESH = 3e6
MIN_AREA = 50
BLOCK = 10000          # frames per contiguous read (~9 GB)


def frame_stats(data):
    mask = data >= THRESH
    area = int(mask.sum())
    if area < MIN_AREA:
        return np.nan, np.nan, np.nan, np.nan, area
    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        mask = labels == (np.argmax(sizes) + 1)
        area = int(mask.sum())
    ys, xs = np.nonzero(mask)
    wt = data[ys, xs] - THRESH
    W = wt.sum()
    cx = (xs * wt).sum() / W
    cy = (ys * wt).sum() / W
    dx, dy = xs - cx, ys - cy
    mu20 = (wt * dx * dx).sum() / W
    mu02 = (wt * dy * dy).sum() / W
    mu11 = (wt * dx * dy).sum() / W
    th = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)
    common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
    lam1 = (mu20 + mu02 + common) / 2
    lam2 = (mu20 + mu02 - common) / 2
    ecc = np.sqrt(max(0.0, 1 - lam2 / lam1)) if lam1 > 0 else 0.0
    return cx, cy, th, ecc, area


with TdmsFile.open(TDMS_PATH) as tdms:
    ch = tdms[GROUP][CHANNEL]
    w = int(ch.properties["Image X"]); h = int(ch.properties["Image Y"])
    fs = w * h
    n_frames = len(ch) // fs
    print(f"Image {w}x{h}; {n_frames} frames total; block={BLOCK}", flush=True)

    cx = np.full(n_frames, np.nan); cy = np.full(n_frames, np.nan)
    th = np.full(n_frames, np.nan); ecc = np.full(n_frames, np.nan)
    area = np.zeros(n_frames, np.int64)

    for start in range(0, n_frames, BLOCK):
        count = min(BLOCK, n_frames - start)
        block = ch.read_data(offset=start * fs, length=count * fs).reshape(count, h, w)
        for i in range(count):
            cx[start+i], cy[start+i], th[start+i], ecc[start+i], area[start+i] = \
                frame_stats(block[i].astype(np.float64))
        del block
        print(f"  processed {start+count}/{n_frames}", flush=True)

np.savez(OUT_DIR / "all_tracks.npz",
         cx=cx, cy=cy, theta=th, ecc=ecc, area=area, n_frames=n_frames)
print(f"Saved: {OUT_DIR / 'all_tracks.npz'}", flush=True)
