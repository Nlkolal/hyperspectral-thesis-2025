from pathlib import Path
import numpy as np
import sys

base = Path("dataset") / 'h2_normal'
dir_target = Path('dataset') / 'h2_normal_norm'

if dir_target.is_dir():
    sys.exit("the target folder is allredy created, write a new")
else:
    dir_target.mkdir()  

train = [0,1,3,5,6,7]
val   = [4]
test  = [2,8]

# ---- fit stats on train ----
X = []
for i in train:
    a = np.load(base / f"data_{i}.npy").astype(np.float32)
    H,W,B = a.shape
    X.append(a.reshape(-1,B))
X = np.vstack(X)
mu = X.mean(0).astype(np.float32)
sd = (X.std(0) + 1e-6).astype(np.float32)
np.save(dir_target / "mu.npy", mu)
np.save(dir_target / "sd.npy", sd)

def norm_and_save(ids):
    for i in ids:
        a = np.load(base / f"data_{i}.npy").astype(np.float32)
        a = (a - mu) / sd
        np.save(dir_target / f"data_{i}.npy", a)
        y = np.load(base / f"label_{i}.npy")
        np.save(dir_target / f"label_{i}.npy", y)

norm_and_save(train)
norm_and_save(val)
norm_and_save(test)

# sanity check on train after normalization
chk = []
for i in train:
    a = np.load(dir_target / f"data_{i}.npy")
    H,W,B = a.shape
    chk.append(a.reshape(-1,B))
chk = np.vstack(chk)
print("train mean (≈0):", chk.mean(0)[:5])
print("train std  (≈1):", chk.std(0)[:5])
print("done")

print("done")