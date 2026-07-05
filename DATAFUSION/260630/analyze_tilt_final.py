"""Final tilt estimate from all-frame cached tracks (no I/O).

Every layer prints the same square, so pooled per-frame tail-offset vectors form
4 clusters (the 4 travel dirs, 90 deg apart) rotated by the tilt theta, on top of
a constant bias.  We fit (bias, theta) by maximizing the bias-removed 4-fold order
parameter over a small grid of bias, then read theta from the phase.
Uncertainty = spread of theta across the 10 layers.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "all_tracks.npz")
cx, cy, area = d["cx"], d["cy"], d["area"]
N = int(d["n_frames"])
fi = np.arange(N)
valid = (area >= 50) & ~np.isnan(cx)
print(f"valid frames: {valid.sum()}/{N}")

gx, gy = np.median(cx[valid]), np.median(cy[valid])
ox = cx - gx; oy = cy - gy
mag = np.hypot(ox, oy)

MAGMIN = 3.0          # require a clear tail; drops near-corner / round frames
use = valid & (mag > MAGMIN)
zx, zy = ox[use], oy[use]
print(f"frames with |offset|>{MAGMIN}: {use.sum()}")


def order4(bx, by):
    """bias-removed 4-fold order parameter and phase over the used frames."""
    rx, ry = zx - bx, zy - by
    r = np.hypot(rx, ry)
    ok = r > 1e-6
    ang = np.arctan2(ry[ok], rx[ok])
    S = np.mean(np.exp(4j * ang))     # unit weight -> magnitude-imbalance-proof
    return np.abs(S), np.angle(S)


# grid-search bias to maximize 4-fold concentration
gs = np.arange(-8, 8.01, 0.5)
bestR, bestb, bestphase = -1, (0, 0), 0
for bx in gs:
    for by in gs:
        R, ph = order4(bx, by)
        if R > bestR:
            bestR, bestb, bestphase = R, (bx, by), ph
theta = np.degrees(bestphase) / 4.0
# fold to (-22.5, 22.5]
theta = (theta + 22.5) % 45 - 22.5
print(f"\n=== POOLED FIT (all layers) ===")
print(f"bias vector: dx={bestb[0]:+.2f} dy={bestb[1]:+.2f} px")
print(f"4-fold concentration R = {bestR:.3f}")
print(f"TILT theta = {theta:+.3f} deg  (clockwise +, image axes)")

# ---- per-layer breakdown for uncertainty ----
# split the build into 10 layers by equal division of the active-print span
active = np.where(valid)[0]
f0, f1 = active[0], active[-1]
edges = np.linspace(f0, f1 + 1, 11).astype(int)
bx, by = bestb
print(f"\nper-layer tilt (bias fixed at pooled value):")
layer_thetas = []
for L in range(10):
    m = use & (fi >= edges[L]) & (fi < edges[L + 1])
    if m.sum() < 50:
        continue
    rx, ry = ox[m] - bx, oy[m] - by
    ang = np.arctan2(ry, rx)
    S = np.mean(np.exp(4j * ang))
    tL = np.degrees(np.angle(S)) / 4.0
    tL = (tL + 22.5) % 45 - 22.5
    layer_thetas.append(tL)
    print(f"  layer {L} frames {edges[L]:5d}-{edges[L+1]:5d}: "
          f"theta={tL:+6.2f} deg  R={np.abs(S):.2f}  n={m.sum()}")

layer_thetas = np.array(layer_thetas)
print(f"\nlayer mean {layer_thetas.mean():+.3f} deg, std {layer_thetas.std():.3f} deg, "
      f"SEM {layer_thetas.std()/np.sqrt(len(layer_thetas)):.3f} deg")

# ---- plot: pooled offset scatter + fitted cross ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
sc = ax[0].scatter(ox[use] - bx, oy[use] - by, c=fi[use], cmap="twilight", s=3, alpha=0.3)
th_r = np.radians(theta)
for k in range(4):
    a = th_r + k * np.pi / 2
    ax[0].plot([0, 25 * np.cos(a)], [0, 25 * np.sin(a)], "r-", lw=2)
ax[0].axhline(0, color="k", lw=0.4); ax[0].axvline(0, color="k", lw=0.4)
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].set_xlabel("offset x (bias-removed) [px]"); ax[0].set_ylabel("offset y [px]")
ax[0].set_title(f"pooled tail offsets + fitted 4-fold cross\ntheta = {theta:+.2f} deg")
fig.colorbar(sc, ax=ax[0], label="frame")

ax[1].plot(range(len(layer_thetas)), layer_thetas, "o-")
ax[1].axhline(layer_thetas.mean(), color="r", ls="--",
              label=f"mean {layer_thetas.mean():+.2f}")
ax[1].fill_between(range(len(layer_thetas)),
                   layer_thetas.mean() - layer_thetas.std(),
                   layer_thetas.mean() + layer_thetas.std(),
                   color="r", alpha=0.15, label=f"+/-std {layer_thetas.std():.2f}")
ax[1].set_xlabel("layer"); ax[1].set_ylabel("tilt [deg]")
ax[1].set_title("per-layer tilt (consistency check)"); ax[1].legend()

fig.suptitle(f"Melt-pool toolpath tilt = {theta:+.2f} deg  "
             f"(pooled R={bestR:.2f}, layer std {layer_thetas.std():.2f} deg)", fontsize=13)
fig.tight_layout()
out = OUT_DIR / "tilt_final.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
