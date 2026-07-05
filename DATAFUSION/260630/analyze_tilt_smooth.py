"""See the whole-build tail-direction structure, and estimate tilt from the
SMOOTHED offset (kills per-frame jitter; tail is ~constant for ~1100 frames/side).
Cached tracks, no I/O.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "all_tracks.npz")
cx, cy, area = d["cx"], d["cy"], d["area"]
N = int(d["n_frames"]); fi = np.arange(N)
valid = (area >= 50) & ~np.isnan(cx)
gx, gy = np.median(cx[valid]), np.median(cy[valid])

# work on the ordered valid subsequence
iv = np.where(valid)[0]
fx = fi[iv]
ox = cx[iv] - gx; oy = cy[iv] - gy

# smooth within-side (window ~151 valid frames << 1100/side, >> jitter)
W = 151
oxs = uniform_filter1d(ox, W, mode="nearest")
oys = uniform_filter1d(oy, W, mode="nearest")
mags = np.hypot(oxs, oys)


def order4(bx, by, X, Y, wt=None):
    rx, ry = X - bx, Y - by
    r = np.hypot(rx, ry); ok = r > 1e-6
    ang = np.arctan2(ry[ok], rx[ok])
    w = (r[ok] if wt is None else wt[ok])
    S = np.sum(w * np.exp(4j * ang)) / np.sum(w)
    return np.abs(S), np.angle(S)


# grid-search bias on smoothed offsets, magnitude-weighted, require clear tail
usem = mags > 4.0
best = (-1, 0, 0, 0)
for bx in np.arange(-8, 8.01, 0.5):
    for by in np.arange(-8, 8.01, 0.5):
        R, ph = order4(bx, by, oxs[usem], oys[usem])
        if R > best[0]:
            best = (R, bx, by, ph)
R, bx, by, ph = best
theta = (np.degrees(ph)/4 + 22.5) % 45 - 22.5
print(f"bias dx={bx:+.2f} dy={by:+.2f}  R={R:.3f}")
print(f"TILT theta = {theta:+.3f} deg  (smoothed, all layers)")

# per-layer via detected repeats: report tilt in 10 equal chunks of valid seq
chunks = np.array_split(np.arange(len(fx)), 10)
per = []
for c in chunks:
    m = c[mags[c] > 4.0]
    _, phc = order4(bx, by, oxs[m], oys[m])
    per.append((np.degrees(phc)/4 + 22.5) % 45 - 22.5)
per = np.array(per)
print(f"per-chunk tilt: {np.round(per,2)}")
print(f"chunk mean {per.mean():+.2f}, std {per.std():.2f} deg")

# ---- plots ----
fig, ax = plt.subplots(2, 1, figsize=(15, 9))
ax[0].plot(fx, oxs, lw=1, label="offset x (smoothed)")
ax[0].plot(fx, oys, lw=1, label="offset y (smoothed)")
ax[0].axhline(0, color="k", lw=0.4)
ax[0].set_ylabel("offset [px]"); ax[0].legend(loc="upper right")
ax[0].set_title("Smoothed tail offset over whole build (staircase = 4 travel dirs, repeating per layer)")

ang_s = np.degrees(np.arctan2(oys - by, oxs - bx))
ax[1].plot(fx[usem], ang_s[usem], ".", ms=1)
ax[1].set_ylabel("tail direction [deg]"); ax[1].set_xlabel("frame")
ax[1].set_title("Tail direction vs frame (bias-removed)")
fig.tight_layout()
out = OUT_DIR / "tilt_staircase.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"Saved: {out}")

# offset-space scatter colored by frame + fitted cross
fig2, a2 = plt.subplots(figsize=(7, 7))
sc = a2.scatter(oxs[usem]-bx, oys[usem]-by, c=fx[usem], cmap="twilight", s=2, alpha=0.4)
for k in range(4):
    aa = np.radians(theta) + k*np.pi/2
    a2.plot([0, 15*np.cos(aa)], [0, 15*np.sin(aa)], "r-", lw=2)
a2.axhline(0, color="k", lw=0.4); a2.axvline(0, color="k", lw=0.4)
a2.set_aspect("equal"); a2.invert_yaxis()
a2.set_xlabel("offset x [px]"); a2.set_ylabel("offset y [px]")
a2.set_title(f"Smoothed tail offsets (all layers)  theta={theta:+.2f} deg  R={R:.2f}")
fig2.colorbar(sc, ax=a2, label="frame")
fig2.tight_layout()
out2 = OUT_DIR / "tilt_staircase_scatter.png"
fig2.savefig(out2, dpi=120, bbox_inches="tight")
print(f"Saved: {out2}")
