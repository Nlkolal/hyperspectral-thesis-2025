from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

base = Path("dataset") / 'h2_normal'

# guess RGB band indices (tweak later)
iR, iG, iB = 65, 45, 25

def to_rgb(img):
    R = img[..., iR]
    G = img[..., iG]
    B = img[..., iB]
    rgb = np.stack([R, G, B], axis=-1).astype(np.float32)
    # simple per-channel 2–98% stretch
    for c in range(3):
        p2, p98 = np.percentile(rgb[..., c], [2, 98])
        if p98 > p2: rgb[..., c] = (rgb[..., c] - p2) / (p98 - p2)
    return np.clip(rgb, 0, 1)

# --- RGBs ---
fig, axes = plt.subplots(3, 3, figsize=(10,10))
axes = axes.ravel()
for i in range(9):
    data = np.load(base / f"data_{i}.npy")
    axes[i].imshow(to_rgb(data))
    axes[i].set_title(f"data_{i}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()

# --- labels ---
fig, axes = plt.subplots(3, 3, figsize=(10,10))
axes = axes.ravel()
for i in range(9):
    lab = np.load(base / f"label_{i}.npy")
    print(f"Unique values in label{i}: {np.unique(lab, return_counts=True)}")
    axes[i].imshow(lab, interpolation="nearest")
    axes[i].set_title(f"label_{i}")
    axes[i].axis("off")
plt.tight_layout()
plt.show()


