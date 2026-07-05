"""Tilt from 40 side-means (4 sides x 10 layers).  Cached tracks, no I/O.

Per-frame offset is too noisy (R~0.13). Averaging each side (~1100 frames)
recovers the 4-fold signal. Every layer repeats the same square, so side s and
side s+4 share a travel direction -> we just segment the whole active span into
~4-side windows, collect all side-mean tail vectors, remove the common bias, and
fold-4 their directions.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "all_tracks.npz")
cx, cy, area = d["cx"], d["cy"], d["area"]
N = int(d["n_frames"]); fi = np.arange(N)
valid = (area >= 50) & ~np.isnan(cx)
gx, gy = np.median(cx[valid]), np.median(cy[valid])
ox, oy = cx - gx, cy - gy


def segment4(x, y):
    """DP: split ordered pts into 4 variance-min segments -> 3 breakpoints."""
    M = len(x)
    c1x = np.concatenate([[0], np.cumsum(x)]); c2x = np.concatenate([[0], np.cumsum(x*x)])
    c1y = np.concatenate([[0], np.cumsum(y)]); c2y = np.concatenate([[0], np.cumsum(y*y)])
    def sse(a, b):
        n = b - a
        if n <= 0: return 0.0
        sx = c1x[b]-c1x[a]; sy = c1y[b]-c1y[a]
        return (c2x[b]-c2x[a]-sx*sx/n) + (c2y[b]-c2y[a]-sy*sy/n)
    K = 4
    dp = np.full((K+1, M+1), np.inf); bk = np.zeros((K+1, M+1), int); dp[0][0] = 0
    for k in range(1, K+1):
        for b in range(k, M+1):
            best, arg = np.inf, -1
            for a in range(k-1, b):
                if dp[k-1][a] == np.inf: continue
                c = dp[k-1][a] + sse(a, b)
                if c < best: best, arg = c, a
            dp[k][b], bk[k][b] = best, arg
    bnds = [M]; b = M
    for k in range(K, 0, -1):
        b = bk[k][b]; bnds.append(b)
    return sorted(bnds)


# active-print span, divided into 10 rough layer windows (exact boundaries not needed)
active = np.where(valid)[0]
edges = np.linspace(active[0], active[-1]+1, 11).astype(int)

side_vecs = []          # (dx, dy) per side
side_layer = []
for L in range(10):
    m = valid & (fi >= edges[L]) & (fi < edges[L+1])
    idxL = np.where(m)[0]
    xs, ys = ox[idxL], oy[idxL]
    # subsample for DP speed
    step = max(1, len(xs) // 900)
    xss, yss = xs[::step], ys[::step]
    bnds = segment4(xss, yss)
    for s in range(4):
        a, b = bnds[s], bnds[s+1]
        side_vecs.append((xss[a:b].mean(), yss[a:b].mean()))
        side_layer.append(L)

side_vecs = np.array(side_vecs)          # 40 x 2
side_layer = np.array(side_layer)

# remove common bias (mean of all side vectors ~ optical offset; rotations cancel)
bias = side_vecs.mean(axis=0)
res = side_vecs - bias
res_dir = np.arctan2(res[:, 1], res[:, 0])
res_mag = np.hypot(res[:, 0], res[:, 1])
print(f"common bias: dx={bias[0]:+.2f} dy={bias[1]:+.2f} px   (n_sides={len(res)})")


def fold4(ang, wt):
    S = np.sum(wt * np.exp(4j*ang)) / np.sum(wt)
    t = np.degrees(np.angle(S))/4.0
    return (t + 22.5) % 45 - 22.5, np.abs(S)


theta, R = fold4(res_dir, res_mag)
print(f"\n=== TILT (40 side-means, bias-removed, mag-weighted fold4) ===")
print(f"TILT theta = {theta:+.3f} deg   (R={R:.3f})")

# per-layer tilt for uncertainty (4 sides each)
per = []
for L in range(10):
    mask = side_layer == L
    tL, RL = fold4(res_dir[mask], res_mag[mask])
    per.append(tL)
per = np.array(per)
print(f"\nper-layer tilt: {np.round(per,2)}")
print(f"layer mean {per.mean():+.3f} deg, std {per.std():.3f}, "
      f"SEM {per.std()/np.sqrt(len(per)):.3f} deg")

# bootstrap CI over the 40 sides
rng_idx = np.arange(len(res_dir))
boot = []
for k in range(2000):
    samp = (rng_idx * 7 + k * 13) % len(res_dir)      # deterministic resample
    samp = np.sort(samp)[np.random.default_rng(k).integers(0, len(res_dir), len(res_dir))] \
        if False else np.random.default_rng(k).integers(0, len(res_dir), len(res_dir))
    tb, _ = fold4(res_dir[samp], res_mag[samp])
    boot.append(tb)
boot = np.array(boot)
print(f"bootstrap 95% CI: [{np.percentile(boot,2.5):+.2f}, {np.percentile(boot,97.5):+.2f}] deg")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for i in range(len(res)):
    ax[0].annotate("", xy=(res[i, 0], res[i, 1]), xytext=(0, 0),
                   arrowprops=dict(color=cols[side_layer[i]], width=1, headwidth=6, alpha=0.7))
th_r = np.radians(theta)
for k in range(4):
    a = th_r + k*np.pi/2
    ax[0].plot([0, 15*np.cos(a)], [0, 15*np.sin(a)], "k-", lw=2.5)
ax[0].axhline(0, color="gray", lw=0.4); ax[0].axvline(0, color="gray", lw=0.4)
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].set_xlabel("residual offset x [px]"); ax[0].set_ylabel("residual offset y [px]")
ax[0].set_title(f"40 side-mean tail vectors (bias-removed)\n+ fitted cross theta={theta:+.2f} deg")

ax[1].plot(range(10), per, "o-")
ax[1].axhline(theta, color="r", ls="--", label=f"pooled {theta:+.2f}")
ax[1].fill_between(range(10), theta-per.std(), theta+per.std(), color="r", alpha=0.15)
ax[1].set_xlabel("layer"); ax[1].set_ylabel("tilt [deg]")
ax[1].set_title(f"per-layer tilt (std {per.std():.2f} deg)"); ax[1].legend()

fig.suptitle(f"Toolpath tilt = {theta:+.2f} deg  "
             f"(40 sides, R={R:.2f}, layer std {per.std():.2f} deg)", fontsize=13)
fig.tight_layout()
out = OUT_DIR / "tilt_final2.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
