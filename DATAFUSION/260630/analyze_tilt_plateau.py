"""Tilt from PLATEAU frames only (straight sides), corners removed.
The tail direction ramps ~360 deg/layer (sawtooth); on a straight side it is
stable, at a corner it swings fast. Keep only low-angular-speed frames -> 4 clean
travel directions -> fold-4 -> tilt. Unwrapped direction splits layers cleanly.
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

iv = np.where(valid)[0]
fx = fi[iv]
ox = uniform_filter1d(cx[iv]-gx, 151, mode="nearest")
oy = uniform_filter1d(cy[iv]-gy, 151, mode="nearest")
mag = np.hypot(ox, oy)

# unwrap direction along the ordered sequence; angular speed for plateau detection
raw_ang = np.arctan2(oy, ox)
# only unwrap across small frame gaps
uang = np.copy(raw_ang)
for i in range(1, len(uang)):
    dphi = raw_ang[i] - raw_ang[i-1]
    dphi = (dphi + np.pi) % (2*np.pi) - np.pi
    uang[i] = uang[i-1] + dphi
speed = np.abs(np.gradient(uang))                     # rad/frame
# layer index from unwrapped direction (one +2pi per layer)
layer = np.floor((uang - uang[0]) / (2*np.pi)).astype(int)
n_layers = layer.max() - layer.min() + 1
print(f"detected ~{n_layers} direction wraps (layers) across the build")

# plateau = stable direction (bottom 45% angular speed) AND clear tail
thr = np.percentile(speed, 45)
plateau = (speed < thr) & (mag > 4.0)
print(f"plateau frames: {plateau.sum()} / {len(fx)}")


def order4(bx, by, X, Y):
    rx, ry = X-bx, Y-by
    r = np.hypot(rx, ry); ok = r > 1e-6
    ang = np.arctan2(ry[ok], rx[ok])
    S = np.sum(r[ok]*np.exp(4j*ang))/np.sum(r[ok])
    return np.abs(S), np.angle(S)


# joint bias + tilt on plateau frames
Xp, Yp = ox[plateau], oy[plateau]
best = (-1, 0, 0, 0)
for bx in np.arange(-6, 6.01, 0.4):
    for by in np.arange(-6, 6.01, 0.4):
        R, ph = order4(bx, by, Xp, Yp)
        if R > best[0]:
            best = (R, bx, by, ph)
R, bx, by, ph = best
theta = (np.degrees(ph)/4 + 22.5) % 45 - 22.5
print(f"\n=== PLATEAU FIT ===")
print(f"bias dx={bx:+.2f} dy={by:+.2f}  R={R:.3f}")
print(f"TILT theta = {theta:+.3f} deg")

# per-layer tilt on plateau frames
per = []
for L in range(layer.min(), layer.max()+1):
    m = plateau & (layer == L)
    if m.sum() < 100: continue
    _, phL = order4(bx, by, ox[m], oy[m])
    tL = (np.degrees(phL)/4 + 22.5) % 45 - 22.5
    per.append(tL)
per = np.array(per)
print(f"per-layer tilt: {np.round(per,2)}")
print(f"layer mean {per.mean():+.3f} deg, std {per.std():.3f}, "
      f"SEM {per.std()/np.sqrt(max(1,len(per))):.3f} deg")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
ax[0].scatter(ox[~plateau]-bx, oy[~plateau]-by, s=2, color="lightgray", alpha=0.3, label="corners (excl.)")
sc = ax[0].scatter(Xp-bx, Yp-by, c=fx[plateau], cmap="twilight", s=4, alpha=0.6)
for k in range(4):
    a = np.radians(theta) + k*np.pi/2
    ax[0].plot([0, 15*np.cos(a)], [0, 15*np.sin(a)], "r-", lw=2.5)
ax[0].axhline(0, color="k", lw=0.4); ax[0].axvline(0, color="k", lw=0.4)
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].set_xlabel("offset x [px]"); ax[0].set_ylabel("offset y [px]")
ax[0].set_title(f"plateau tail offsets (sides only)\ntheta = {theta:+.2f} deg  R={R:.2f}")
ax[0].legend(loc="lower right", fontsize=8)
fig.colorbar(sc, ax=ax[0], label="frame")

ax[1].plot(range(len(per)), per, "o-")
ax[1].axhline(theta, color="r", ls="--", label=f"pooled {theta:+.2f}")
ax[1].fill_between(range(len(per)), theta-per.std(), theta+per.std(), color="r", alpha=0.15)
ax[1].set_xlabel("layer"); ax[1].set_ylabel("tilt [deg]")
ax[1].set_title(f"per-layer tilt (std {per.std():.2f} deg)"); ax[1].legend()

fig.suptitle(f"Toolpath tilt = {theta:+.2f} deg  (plateau frames, R={R:.2f}, "
             f"layer std {per.std():.2f} deg)", fontsize=13)
fig.tight_layout()
out = OUT_DIR / "tilt_plateau.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
