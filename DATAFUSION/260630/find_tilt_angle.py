"""Estimate the in-plane tilt angle of the melt-pool images.

One print layer is a square toolpath -> travel direction takes 4 values 90 deg
apart. In the image they appear rotated by the unknown tilt theta. We recover
theta from two independent signals and cross-check:

  (A) centroid translation  : path traces a rotated square (fixed camera)
  (B) pool major-axis angle : elongated tail points along travel (coaxial cam)

Both have 4-fold (90 deg) symmetry, so we fold by x4 and take the
magnitude-weighted circular mean, then divide by 4.  theta in (-22.5, 22.5].
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from nptdms import TdmsFile

TDMS_PATH = Path("/mnt/a/ftk3187/HAMMER/DED/RGB/raw_test/HAMMER_Square_600W_1_12.tdms")
OUT_DIR = Path("/home/ftk3187/github/DPC_research/DATAFUSION/260630/tdms_inspection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

GROUP, CHANNEL = "Image Data", "Flattened Images"
THRESH = 3e6           # melt-pool level
F_START = 1000         # skip initialization phase
F_END = 6250           # ~one layer (5000 frames) + margin
STEP = 1               # every frame (block is read once into RAM)
MIN_AREA = 50          # px; below this -> no pool / dropped frame

with TdmsFile.open(TDMS_PATH) as tdms:
    ch = tdms[GROUP][CHANNEL]
    w = int(ch.properties["Image X"])
    h = int(ch.properties["Image Y"])
    fs = w * h
    n_block = F_END - F_START
    print(f"Image {w}x{h}; reading contiguous block frames {F_START}..{F_END} "
          f"({n_block} frames, ~{n_block*fs*4/1024**3:.1f} GB) in one seek...")
    # ONE sequential read of the whole layer -> avoids per-frame seek cost
    block = ch.read_data(offset=F_START * fs, length=n_block * fs)
    block = block.reshape(n_block, h, w)
    print("  block read done; processing frames in RAM")

    frames_idx = np.arange(F_START, F_END, STEP)
    fi_list, cx_list, cy_list, th_list, ecc_list, area_list = ([] for _ in range(6))
    for k, fi in enumerate(frames_idx):
        data = block[fi - F_START].astype(np.float64)
        mask = data >= THRESH
        area = int(mask.sum())
        cx = cy = th = ecc = np.nan
        if area >= MIN_AREA:
            labels, n = ndimage.label(mask)
            if n > 1:
                sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
                mask = labels == (np.argmax(sizes) + 1)
                area = int(mask.sum())
            ys, xs = np.nonzero(mask)
            wt = data[ys, xs] - THRESH          # weight by excess intensity
            W = wt.sum()
            cx = (xs * wt).sum() / W
            cy = (ys * wt).sum() / W
            dx, dy = xs - cx, ys - cy
            mu20 = (wt * dx * dx).sum() / W
            mu02 = (wt * dy * dy).sum() / W
            mu11 = (wt * dx * dy).sum() / W
            th = 0.5 * np.arctan2(2 * mu11, mu20 - mu02)   # major-axis angle (rad)
            common = np.sqrt(4 * mu11 ** 2 + (mu20 - mu02) ** 2)
            lam1 = (mu20 + mu02 + common) / 2
            lam2 = (mu20 + mu02 - common) / 2
            ecc = np.sqrt(max(0.0, 1 - lam2 / lam1)) if lam1 > 0 else 0.0
        fi_list.append(fi); cx_list.append(cx); cy_list.append(cy)
        th_list.append(th); ecc_list.append(ecc); area_list.append(area)
        if k % 100 == 0:
            print(f"  {k}/{len(frames_idx)} frame {fi} area={area}")

fi_ = np.array(fi_list, float)
cx_ = np.array(cx_list); cy_ = np.array(cy_list)
th_ = np.array(th_list); ecc_ = np.array(ecc_list)
area_ = np.array(area_list, float)
valid = (area_ >= MIN_AREA) & ~np.isnan(cx_)
print(f"valid frames: {valid.sum()}/{len(valid)}")


def fold4_mean(angles_rad, weights):
    """Magnitude-weighted circular mean under 90-deg (4-fold) symmetry."""
    S = np.sum(weights * np.exp(4j * angles_rad))
    theta = np.angle(S) / 4.0
    R = np.abs(S) / np.sum(weights)        # concentration in [0,1]
    return np.degrees(theta), R


# ---- (A) translation: direction of centroid motion between samples ----
dcx, dcy = np.diff(cx_), np.diff(cy_)
seg_ok = valid[:-1] & valid[1:]
mag = np.hypot(dcx, dcy)
useA = seg_ok & (mag > 0.5)
angA = np.arctan2(dcy[useA], dcx[useA])
thetaA, RA = fold4_mean(angA, mag[useA])

# ---- (B) orientation: pool major-axis angle ----
useB = valid & (ecc_ > 0.3)
thetaB, RB = fold4_mean(th_[useB], ecc_[useB])

print("\n=== tilt estimates (fold to +/-22.5 deg) ===")
print(f"(A) centroid-translation : theta = {thetaA:+.3f} deg   (R={RA:.3f}, n={useA.sum()})")
print(f"(B) pool-orientation     : theta = {thetaB:+.3f} deg   (R={RB:.3f}, n={useB.sum()})")

# ---- diagnostics ----
fig, ax = plt.subplots(2, 2, figsize=(13, 11))

sc = ax[0, 0].scatter(cx_[valid], cy_[valid], c=fi_[valid], cmap="viridis", s=8)
ax[0, 0].set_title("(A) centroid trajectory (colored by frame)")
ax[0, 0].set_xlabel("cx [px]"); ax[0, 0].set_ylabel("cy [px]")
ax[0, 0].set_aspect("equal"); ax[0, 0].invert_yaxis()
fig.colorbar(sc, ax=ax[0, 0], label="frame")

ax[0, 1].plot(fi_[valid], cx_[valid], ".", ms=3, label="cx")
ax[0, 1].plot(fi_[valid], cy_[valid], ".", ms=3, label="cy")
ax[0, 1].set_title("centroid vs frame"); ax[0, 1].set_xlabel("frame"); ax[0, 1].legend()

ax[1, 0].plot(fi_[valid], np.degrees(th_[valid]), ".", ms=3)
ax[1, 0].set_title("(B) pool major-axis angle vs frame")
ax[1, 0].set_xlabel("frame"); ax[1, 0].set_ylabel("angle [deg]")

ax[1, 1].plot(fi_[1:][useA], np.degrees(angA), ".", ms=3, alpha=0.5)
ax[1, 1].axhline(0, color="k", lw=0.5)
ax[1, 1].set_title("(A) centroid step direction vs frame")
ax[1, 1].set_xlabel("frame"); ax[1, 1].set_ylabel("dir [deg]")

fig.suptitle(
    f"Tilt: (A) translation {thetaA:+.2f} deg  |  (B) orientation {thetaB:+.2f} deg",
    fontsize=13,
)
fig.tight_layout()
out = OUT_DIR / "tilt_diagnostic.png"
fig.savefig(out, dpi=120, bbox_inches="tight")
np.savez(OUT_DIR / "tilt_tracks.npz",
         fi=fi_, cx=cx_, cy=cy_, theta=th_, ecc=ecc_, area=area_)
print(f"Saved: {out}")
print(f"Saved: {OUT_DIR / 'tilt_tracks.npz'}")
