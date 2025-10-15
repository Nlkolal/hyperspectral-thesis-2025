from pathlib import Path
import numpy as np
from sklearn.decomposition import PCA
import sys

base = Path("dataset") / "h2_normal_norm"
out  = Path("dataset") / "h2_pca6"

if out.exists():
    sys.exit("the target folder is allredy created, write a new")
else:
    out.mkdir()  


train = [0,1,3,5,6,7]
val   = [4]
test  = [2,8]
K = 6  # number of PCA components to keep

# ---- fit PCA on TRAIN (data already normalized) ----
X = []
for i in train:
    a = np.load(base / f"data_{i}.npy").astype(np.float32)
    H,W,B = a.shape
    X.append(a.reshape(-1, B))
X = np.vstack(X)
pca = PCA(n_components=K, whiten=True, random_state=0).fit(X)

# save PCA params
np.save(out / "pca_components.npy", pca.components_.astype(np.float32))
np.save(out / "pca_mean.npy",       pca.mean_.astype(np.float32))
np.save(out / "explained_variance_ratio.npy", pca.explained_variance_ratio_.astype(np.float32))

def transform_and_save(ids):
    for i in ids:
        a = np.load(base / f"data_{i}.npy").astype(np.float32)   # (H,W,B)
        H,W,B = a.shape
        Z = pca.transform(a.reshape(-1, B)).reshape(H, W, K).astype(np.float32)
        np.save(out / f"data_{i}.npy", Z)
        y = np.load(base / f"label_{i}.npy")
        np.save(out / f"label_{i}.npy", y)

transform_and_save(train)
transform_and_save(val)
transform_and_save(test)

# quick print
print("saved PCA dataset to:", out)
print("explained variance ratio (first 10):", np.round(pca.explained_variance_ratio_[:10], 4))
