"""Estimate tilt from cached per-frame tracks (no 42GB re-read).

Signal: the melt pool stays centered (coaxial cam) but its intensity centroid
is pulled toward the cooling tail, which points opposite travel. Over the 4
sides of the square the offset vector (centroid - global center) takes 4
directions 90 deg apart -> fold-4 circular mean gives the tilt theta.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "tilt_tracks.npz")
fi, cx, cy, area = d["fi"], d["cx"], d["cy"], d["area"]
valid = (area >= 50) & ~np.isnan(cx)

# robust global center
gx, gy = np.median(cx[valid]), np.median(cy[valid])
offx, offy = cx - gx, cy - gy
mag = np.hypot(offx, offy)


def fold4_mean(ang, wt):
    S = np.sum(wt * np.exp(4j * ang))
    return np.degrees(np.angle(S) / 4.0), np.abs(S) / np.sum(wt)


# ---- global fold-4 on offset direction ----
use = valid & (mag > 1.0)
ang = np.arctan2(offy[use], offx[use])
thetaC, RC = fold4_mean(ang, mag[use])
print(f"(C) centroid-offset fold4 : theta = {thetaC:+.3f} deg  (R={RC:.3f}, n={use.sum()})")

# ---- per-side validation: split the layer into 4 equal sides in [1000,6000] ----
edges = [1000, 2250, 3500, 4750, 6000]
side_means = []
print("\nper-side mean offset (frames -> dx,dy,dir):")
for s in range(4):
    m = valid & (fi >= edges[s]) & (fi < edges[s + 1])
    dx, dy = offx[m].mean(), offy[m].mean()
    side_means.append((dx, dy))
    print(f"  side {s} [{edges[s]}-{edges[s+1]}]: dx={dx:+6.2f} dy={dy:+6.2f} "
          f"dir={np.degrees(np.arctan2(dy, dx)):+7.2f} deg  (n={m.sum()})")

side_means = np.array(side_means)
sm_ang = np.arctan2(side_means[:, 1], side_means[:, 0])
sm_mag = np.hypot(side_means[:, 0], side_means[:, 1])
thetaS, RS = fold4_mean(sm_ang, sm_mag)
print(f"\n(S) 4-side-mean fold4     : theta = {thetaS:+.3f} deg  (R={RS:.3f})")

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))

sc = ax[0].scatter(offx[valid], offy[valid], c=fi[valid], cmap="viridis", s=6, alpha=0.5)
for s, (dx, dy) in enumerate(side_means):
    ax[0].annotate("", xy=(dx, dy), xytext=(0, 0),
                   arrowprops=dict(color="red", width=2, headwidth=9))
    ax[0].text(dx * 1.15, dy * 1.15, f"side {s}", color="red", fontsize=11, ha="center")
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].set_xlabel("cx - center [px]"); ax[0].set_ylabel("cy - center [px]")
ax[0].set_title(f"centroid offset (tail direction)\nfold4 theta = {thetaC:+.2f} deg (R={RC:.2f})")
fig.colorbar(sc, ax=ax[0], label="frame")

# side assignment vs frame to confirm the 4 segments
ax[1].plot(fi[valid], offx[valid], ".", ms=3, label="cx-center")
ax[1].plot(fi[valid], offy[valid], ".", ms=3, label="cy-center")
for e in edges:
    ax[1].axvline(e, color="gray", ls="--", lw=0.8)
ax[1].set_xlabel("frame"); ax[1].set_ylabel("offset [px]")
ax[1].set_title("centroid offset vs frame (dashed = assumed side splits)")
ax[1].legend()

fig.suptitle(f"Tilt estimate:  centroid-offset fold4 = {thetaC:+.2f} deg   |   "
             f"4-side-mean = {thetaS:+.2f} deg", fontsize=13)
fig.tight_layout()
out = OUT_DIR / "tilt_offset_analysis.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
