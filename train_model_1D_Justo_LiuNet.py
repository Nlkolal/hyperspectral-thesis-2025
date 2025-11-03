# pip install torch torchvision
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random, time, copy

# -----------------
# paths & config
# -----------------
base = Path("dataset/h2_normal")
out  = Path("model/h2_normal"); out.mkdir(parents=True, exist_ok=True)

train_ids, val_ids = [0,1,3,5,6,7], [4, 2, 8]
BATCH = 1024                      # many spectra fit in one batch
EPOCHS = 30
NCLASSES = 3
KERNEL_SIZE = 6                   # per Keras snippet
START_K = 6                       # filters: 6,12,18,24
TRAIN_SAMPLES_PER_EPOCH = 20000   # random pixel spectra per epoch (across train imgs)
VAL_SAMPLES_PER_IMG = 20000       # random per-image for validation; set None to use all pixels
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[device] {device}")

# -----------------
# metrics
# -----------------
def miou_from_preds_gts(pred, gt, ncls=3):
    cm = np.bincount(ncls*gt + pred, minlength=ncls*ncls).reshape(ncls, ncls)
    tp = np.diag(cm); den = cm.sum(1) + cm.sum(0) - tp
    iou = tp / np.maximum(den, 1)
    return float(np.nanmean(iou)), iou

# -----------------
# load one file to get K
# -----------------
K = np.load(base/f"data_{train_ids[0]}.npy").shape[-1]
print(f"[data] spectral features K={K}")

# -----------------
# class weights from *train* distribution (over all pixels)
# -----------------
counts_classes = np.zeros(NCLASSES, dtype=np.int64)
for i in train_ids:
    y = np.load(base / f"label_{i}.npy").ravel()
    for c in range(NCLASSES):
        counts_classes[c] += (y == c).sum()
weights = counts_classes.sum() / (len(counts_classes) * np.maximum(counts_classes, 1))
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)
print("[weights] class counts:", counts_classes.tolist(), " -> weights:", weights.tolist())

# -----------------
# Datasets: return one spectrum per sample: x:[1,K], y:int
# -----------------
class PixelSpectraTrainDS(Dataset):
    """Random pixel sampler from training images."""
    def __init__(self, ids, samples_per_epoch: int):
        self.ids = ids
        self.X = [np.load(base/f"data_{i}.npy").astype(np.float32) for i in ids]
        self.Y = [np.load(base/f"label_{i}.npy").astype(np.int64)  for i in ids]
        self.samples = samples_per_epoch
        H,W,_ = self.X[0].shape
        print(f"[train-ds] {len(ids)} images, sample HxW≈{H}x{W}, spectra/epoch={self.samples}")

    def __len__(self): return self.samples

    def __getitem__(self, _):
        img_idx = random.randrange(len(self.X))
        x = self.X[img_idx]; y = self.Y[img_idx]
        H, W, K = x.shape
        i = random.randrange(H); j = random.randrange(W)
        spec = x[i, j, :]                     # (K,)
        lbl  = y[i, j].item()                 # int
        # Optionally add tiny gaussian noise or spectral jitter here if you want
        return torch.from_numpy(spec[None, :]), torch.tensor(lbl, dtype=torch.long)
        # returns x:[1,K], y:[]

class PixelSpectraValDS(Dataset):
    """Validation: either all pixels (per image) or random subset per image."""
    def __init__(self, ids, samples_per_img=None, seed=0):
        self.ids = ids
        self.X = [np.load(base/f"data_{i}.npy").astype(np.float32) for i in ids]
        self.Y = [np.load(base/f"label_{i}.npy").astype(np.int64)  for i in ids]
        self.indices = []  # list of (img_idx, i, j)
        rng = np.random.default_rng(seed)
        total = 0
        for img_idx, (x, y) in enumerate(zip(self.X, self.Y)):
            H, W, K = x.shape
            if samples_per_img is None or samples_per_img >= H*W:
                # use all pixels
                for ii in range(H):
                    for jj in range(W):
                        self.indices.append((img_idx, ii, jj))
            else:
                # random subset (unique)
                idxs = rng.choice(H*W, size=samples_per_img, replace=False)
                for flat in idxs:
                    ii = int(flat // W); jj = int(flat % W)
                    self.indices.append((img_idx, ii, jj))
            total += (H*W if samples_per_img is None else min(samples_per_img, H*W))
        print(f"[val-ds] {len(ids)} images, total spectra={len(self.indices)}")

    def __len__(self): return len(self.indices)

    def __getitem__(self, k):
        img_idx, i, j = self.indices[k]
        spec = self.X[img_idx][i, j, :]       # (K,)
        lbl  = self.Y[img_idx][i, j].item()
        return torch.from_numpy(spec[None, :]), torch.tensor(lbl, dtype=torch.long)

train_ds = PixelSpectraTrainDS(train_ids, TRAIN_SAMPLES_PER_EPOCH)
val_ds   = PixelSpectraValDS(val_ids, samples_per_img=VAL_SAMPLES_PER_IMG)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                      num_workers=0, pin_memory=(device.type=="cuda"))
val_dl   = DataLoader(val_ds,   batch_size=BATCH, shuffle=False,
                      num_workers=0, pin_memory=(device.type=="cuda"))

# -----------------
# 1D Justo–Liu spectral CNN (no wrapper; direct [N,1,K])
# -----------------
class LiuNet1DSpectral(nn.Module):
    """
    Conv1D(N) + MaxPool(2)
    Conv1D(2*N) + MaxPool(2)
    Conv1D(3*N) + MaxPool(2)
    Conv1D(4*N) + MaxPool(2)
    Flatten -> Dense(num_classes)

    Input:  [N, 1, K]
    """
    def __init__(self, num_features: int, num_classes: int,
                 kernel_size: int = 6, start_k: int = 6):
        super().__init__()
        pad = 2  # 'same'-ish padding so convs don't shrink length

        self.conv1 = nn.Conv1d(1, 1*start_k, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(1*start_k, 2*start_k, kernel_size=kernel_size, padding='same')
        self.conv3 = nn.Conv1d(2*start_k, 3*start_k, kernel_size=kernel_size, padding='same')
        self.conv4 = nn.Conv1d(3*start_k, 4*start_k, kernel_size=kernel_size, padding='same')

        self.relu  = nn.ReLU(inplace=True)
        self.pool  = nn.MaxPool1d(2)

        # spectral length after 4 pools
        L = num_features
        for _ in range(4): L //= 2
        if L < 1:
            raise ValueError(
                f"Spectral features too small ({num_features}) for 4 pools. "
                f"Increase K or reduce pooling depth."
            )
        self.final_len = L
        self.final_ch  = 4*start_k
        self.fc = nn.Linear(self.final_ch * self.final_len, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, 1, K]
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.pool(self.relu(self.conv4(x)))
        x = x.flatten(1)          # [N, final_ch * final_len]
        x = self.fc(x)            # [N, num_classes]
        return x

# -----------------
# build model
# -----------------
model = LiuNet1DSpectral(
    num_features=K,
    num_classes=NCLASSES,
    kernel_size=KERNEL_SIZE,
    start_k=START_K
).to(device)

# -----------------
# training setup
# -----------------
params = sum(p.numel() for p in model.parameters())
print(f"[model] LiuNet1DSpectral(K={K}, ncls={NCLASSES}), params={params/1e6:.2f}M")
print(f"[train] device={device}, BATCH={BATCH}, EPOCHS={EPOCHS}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
lossfn = nn.CrossEntropyLoss(weight=class_weights)

best = {
    "miou": -1.0,
    "epoch": 0,
    "state": None,
    "iou_per_class": None,
}

# -----------------
# train loop
# -----------------
for epoch in range(1, EPOCHS+1):
    t0 = time.time()
    model.train()
    running = 0.0
    for b,(xb,yb) in enumerate(train_dl, 1):
        # xb: [B, 1, K], yb: [B]
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)              # [B, C]
        loss = lossfn(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item()
        if b % 50 == 0:
            print(f"  [epoch {epoch}] batch {b}: loss={loss.item():.4f}")

    # validation
    model.eval()
    total_pred, total_gt = [], []
    with torch.no_grad():
        for xb,yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1).cpu().numpy()
            gt = yb.cpu().numpy()
            total_pred.append(pred); total_gt.append(gt)
    pred = np.concatenate(total_pred)
    gt = np.concatenate(total_gt)
    dt = time.time() - t0

    miou, iou_per_class = miou_from_preds_gts(pred, gt, ncls=NCLASSES)
    print(f"Epoch: {epoch:02d}  mIoU: {miou:.3f}  IoU/cls: {np.round(iou_per_class,3)}  time: {dt:.1f}s")

    if miou > best["miou"]:
        best["miou"] = miou
        best["epoch"] = epoch
        best["state"] = copy.deepcopy(model.state_dict())
        best["iou_per_class"] = iou_per_class.copy()
        print(f"  [best] new best IoU[Cloud, Land, Sea]: {miou:.3f} at epoch {epoch}")

# -----------------
# end-of-run confirm save
# -----------------
print("\nTraining complete.")
print(f"Best mIoU: {best['miou']:.3f} at epoch {best['epoch']}  IoU[Cloud, Land, Sea]: {np.round(best['iou_per_class'],3)}")

ans = input("Will you save this model? (yes/no): ").strip().lower()
if ans in {"y", "yes"} and best["state"] is not None:
    tag = f"e{best['epoch']}_miou{best['miou']:.3f}".replace('.', '-')
    ckpt_path = out / f"liunet1d_{tag}.pt"
    torch.save({
        "model": "LiuNet1DSpectral",
        "state_dict": best["state"],
        "epoch": best["epoch"],
        "miou": best["miou"],
        "iou_per_class": best["iou_per_class"],
        "config": {
            "EPOCHS": EPOCHS, "NCLASSES": NCLASSES,
            "train_ids": train_ids, "val_ids": val_ids,
            "K": K, "kernel_size": KERNEL_SIZE, "start_k": START_K,
            "train_samples_per_epoch": TRAIN_SAMPLES_PER_EPOCH,
            "val_samples_per_img": VAL_SAMPLES_PER_IMG
        }
    }, ckpt_path)
    print(f"[save] saved best model to: {ckpt_path}")
else:
    print("[save] skipped saving model.")
