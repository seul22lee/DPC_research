"""Cross-check tilt from two independent per-side features (cached, no I/O):
  (1) centroid-offset direction  (tail = -travel)
  (2) pool major-axis orientation (elongation = travel axis)
Both averaged per side, then folded to the tilt.  Agreement => confidence.
"""
from pathlib import Path
import numpy as np

OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
d = np.load(OUT_DIR / "tilt_tracks.npz")
fi, cx, cy, th, ecc, area = d["fi"], d["cx"], d["cy"], d["theta"], d["ecc"], d["area"]

LAYER = (fi >= 1000) & (fi < 5400)
valid = LAYER & (area >= 50) & ~np.isnan(cx)
f = fi[valid]
gx, gy = np.median(cx[valid]), np.median(cy[valid])
ox, oy = cx[valid]-gx, cy[valid]-gy
thv, eccv = th[valid], ecc[valid]

# detected side boundaries from v2 (balanced 4 sides)
bnd_frames = [1021, 2072, 3184, 4288, 5366]
idx = [np.searchsorted(f, bf) for bf in bnd_frames]

def foldN_mean(ang, wt, N):
    S = np.sum(wt*np.exp(1j*N*ang))
    return np.angle(S)/N, np.abs(S)/np.sum(wt)

off_dirs, axis_dirs = [], []
print("side | frames        |  tail-dir  |  major-axis  | mean-ecc")
for s in range(4):
    a, b = idx[s], idx[s+1]
    dx, dy = ox[a:b].mean(), oy[a:b].mean()
    od = np.arctan2(dy, dx)
    off_dirs.append(od)
    # major-axis is a director (mod 180) -> circular mean via fold-2, weight by ecc
    ax, _ = foldN_mean(thv[a:b], eccv[a:b], 2)
    axis_dirs.append(ax)
    print(f"  {s}  | {int(f[a]):5d}-{int(f[b-1]):5d}  | {np.degrees(od):+7.2f}   |"
          f" {np.degrees(ax):+7.2f}     | {eccv[a:b].mean():.3f}")

off_dirs = np.array(off_dirs); axis_dirs = np.array(axis_dirs)

# tilt from tail direction: 4 dirs 90 apart -> fold-4
t_off, R_off = foldN_mean(off_dirs, np.ones(4), 4)
# tilt from major axis: director rotates 90/side -> fold-4 as well
t_ax, R_ax = foldN_mean(axis_dirs, np.ones(4), 4)

def wrap(a):  # fold to (-45,45]
    return (np.degrees(a) + 45) % 90 - 45

print("\n=== tilt estimates (fold to +/-22.5 deg) ===")
print(f"(1) tail-offset    : theta = {np.degrees(t_off)/1:+.2f} deg   (R={R_off:.3f})")
print(f"(2) major-axis     : theta = {np.degrees(t_ax):+.2f} deg   (R={R_ax:.3f})")

# also report axis pairing check: side0||side2, side1||side3
print("\naxis pairing (should be ~equal within a pair, pairs ~90 apart):")
print(f"  side0={np.degrees(axis_dirs[0]):+.1f}  side2={np.degrees(axis_dirs[2]):+.1f}"
      f"  |  side1={np.degrees(axis_dirs[1]):+.1f}  side3={np.degrees(axis_dirs[3]):+.1f}")
