"""Accurate tilt via change-point-detected sides (cached tracks, no I/O).

1. per-frame centroid offset = (cx,cy) - global center  (points along tail = -travel)
2. DP change-point detection splits the layer into 4 contiguous sides
3. per-side MEAN offset direction (equal weight per side) -> 4 dirs ~90 apart
4. fold-4 circular mean of those 4 dirs -> tilt theta   (robust to magnitude imbalance)
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "tilt_tracks.npz")
fi, cx, cy, area = d["fi"], d["cx"], d["cy"], d["area"]

# restrict to one layer and valid frames.
# first layer ends at the dropped-frame gap ~5400 (frames after that = next layer)
LAYER = (fi >= 1000) & (fi < 5400)
valid = LAYER & (area >= 50) & ~np.isnan(cx)
f = fi[valid]
gx, gy = np.median(cx[valid]), np.median(cy[valid])
ox, oy = cx[valid] - gx, cy[valid] - gy

# ---- DP change-point: partition ordered points into K=4 variance-min segments ----
K = 4
x, y = ox, oy
M = len(x)
# prefix sums for O(1) segment SSE
c1x = np.concatenate([[0], np.cumsum(x)]);  c2x = np.concatenate([[0], np.cumsum(x*x)])
c1y = np.concatenate([[0], np.cumsum(y)]);  c2y = np.concatenate([[0], np.cumsum(y*y)])

def sse(a, b):
    n = b - a
    if n <= 0:
        return 0.0
    sx = c1x[b]-c1x[a]; sy = c1y[b]-c1y[a]
    return (c2x[b]-c2x[a] - sx*sx/n) + (c2y[b]-c2y[a] - sy*sy/n)

INF = np.inf
# dp[k][b] = min cost of splitting first b points into k segments
dp = np.full((K+1, M+1), INF)
bk = np.zeros((K+1, M+1), int)
dp[0][0] = 0.0
for k in range(1, K+1):
    for b in range(k, M+1):
        best, arg = INF, -1
        for a in range(k-1, b):
            if dp[k-1][a] == INF:
                continue
            c = dp[k-1][a] + sse(a, b)
            if c < best:
                best, arg = c, a
        dp[k][b], bk[k][b] = best, arg

# backtrack breakpoints
bnds = [M]
b = M
for k in range(K, 0, -1):
    a = bk[k][b]
    bnds.append(a); b = a
bnds = sorted(bnds)                       # [0, b1, b2, b3, M]
print("side boundaries (frame):", [int(f[min(i, M-1)]) for i in bnds])


def fold4(ang, wt):
    S = np.sum(wt * np.exp(4j*ang))
    return np.degrees(np.angle(S)/4.0), np.abs(S)/np.sum(wt)


side_dirs, side_mags = [], []
print("\nper-side (equal-weight) mean offset:")
for s in range(K):
    a, b = bnds[s], bnds[s+1]
    dx, dy = x[a:b].mean(), y[a:b].mean()
    ang = np.arctan2(dy, dx); mag = np.hypot(dx, dy)
    side_dirs.append(ang); side_mags.append(mag)
    print(f"  side {s} frames {int(f[a])}-{int(f[b-1])}: dx={dx:+6.2f} dy={dy:+6.2f} "
          f"|off|={mag:5.2f}  dir={np.degrees(ang):+7.2f} deg  (n={b-a})")

side_dirs = np.array(side_dirs)
# check spacing between consecutive side directions
spac = np.degrees(np.diff(np.unwrap(np.sort(side_dirs))))
print(f"\nsorted-dir spacing (deg): {np.round(spac,1)}  (ideal ~90 each)")

theta_eq, R_eq = fold4(side_dirs, np.ones(K))          # equal weight per side
print(f"\n(S*) 4-side equal-weight fold4 : theta = {theta_eq:+.3f} deg  (R={R_eq:.3f})")

# ---- bias-removed: subtract common offset (mean of 4 side vectors) ----
svec = np.array([[np.cos(a)*m, np.sin(a)*m] for a, m in zip(side_dirs, side_mags)])
bias = svec.mean(axis=0)
res = svec - bias
res_dir = np.arctan2(res[:, 1], res[:, 0])
res_mag = np.hypot(res[:, 0], res[:, 1])
print(f"\nestimated bias vector: dx={bias[0]:+.2f} dy={bias[1]:+.2f}")
print("bias-removed per-side dir:", np.round(np.degrees(res_dir), 1),
      " mags:", np.round(res_mag, 1))
spac_r = np.degrees(np.diff(np.unwrap(np.sort(res_dir))))
print(f"bias-removed spacing (deg): {np.round(spac_r,1)}")
theta_br, R_br = fold4(res_dir, np.ones(K))
print(f"(BR) bias-removed 4-side fold4 : theta = {theta_br:+.3f} deg  (R={R_br:.3f})")

# ---- plot ----
fig, ax = plt.subplots(1, 2, figsize=(14, 6.5))
colors = ["tab:purple", "tab:blue", "tab:green", "tab:orange"]
for s in range(K):
    a, b = bnds[s], bnds[s+1]
    ax[0].scatter(x[a:b], y[a:b], s=6, alpha=0.35, color=colors[s], label=f"side {s}")
    dx, dy = x[a:b].mean(), y[a:b].mean()
    ax[0].annotate("", xy=(dx, dy), xytext=(0, 0),
                   arrowprops=dict(color=colors[s], width=2.5, headwidth=10))
ax[0].axhline(0, color="k", lw=0.5); ax[0].axvline(0, color="k", lw=0.5)
ax[0].set_aspect("equal"); ax[0].invert_yaxis()
ax[0].set_xlabel("cx - center [px]"); ax[0].set_ylabel("cy - center [px]")
ax[0].set_title(f"per-side tail direction\nequal-weight fold4 theta = {theta_eq:+.2f} deg")
ax[0].legend(loc="upper right", fontsize=8)

ax[1].plot(f, x, ".", ms=3, label="cx-center")
ax[1].plot(f, y, ".", ms=3, label="cy-center")
for i in bnds[1:-1]:
    ax[1].axvline(f[i], color="red", ls="--", lw=1.2)
ax[1].set_xlabel("frame"); ax[1].set_ylabel("offset [px]")
ax[1].set_title("offset vs frame (red = detected side splits)")
ax[1].legend()

fig.suptitle(f"Tilt angle = {theta_eq:+.2f} deg   (equal-weight 4-side fold4, R={R_eq:.2f})",
             fontsize=13)
fig.tight_layout()
out = OUT_DIR / "tilt_v2_analysis.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
print(f"\nSaved: {out}")
