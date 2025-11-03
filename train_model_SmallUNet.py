# pip install torch torchvision
from pathlib import Path
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random, time, copy

# -----------------
# paths & config
# -----------------
base = Path("dataset/h2_pca16")
out  = Path("model/h2_pca16"); out.mkdir(parents=True, exist_ok=True)

train_ids, val_ids = [0,1,3,5,6,7], [4, 2, 8]
PATCH = 64
BATCH = 64
EPOCHS = 20
NCLASSES = 3
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
# class weights from train distribution
# -----------------
counts_classes = np.zeros(NCLASSES, dtype=np.int64)
for i in train_ids:
    y = np.load(base / f"label_{i}.npy").ravel()
    for c in range(NCLASSES):
        counts_classes[c] += (y == c).sum()
weights = counts_classes.sum() / (len(counts_classes) * np.maximum(counts_classes, 1))
class_weights = torch.tensor(weights, dtype=torch.float32, device=device)

print("[init] loading data indices...")
print("  train:", train_ids)
print("  val  :", val_ids)

# -----------------
# dataset
# -----------------
class PatchDS(Dataset):
    def __init__(self, ids):
        self.X = [np.load(base/f"data_{i}.npy").astype(np.float32) for i in ids]
        self.Y = [np.load(base/f"label_{i}.npy").astype(np.int64)  for i in ids]
        H,W,K = self.X[0].shape
        print(f"[dataset] {len(ids)} files, sample shape: (H={H}, W={W}, K={K})")
    def __len__(self): return 500
    def __getitem__(self, _):
        idx = random.randrange(len(self.X))
        x, y = self.X[idx], self.Y[idx]          # x:(H,W,K)
        H,W,K = x.shape
        i = random.randrange(H-PATCH); j = random.randrange(W-PATCH)
        xp = x[i:i+PATCH, j:j+PATCH, :].transpose(2,0,1)  # (K,H,W)
        yp = y[i:i+PATCH, j:j+PATCH]                      # (H,W)
        if random.random() < 0.5:  # flip W
            xp = np.flip(xp, -1).copy();  yp = np.flip(yp, -1).copy()
        if random.random() < 0.5:  # flip H
            xp = np.flip(xp, -2).copy();  yp = np.flip(yp, -2).copy()
        k = random.randrange(4)     # rotate 0,90,180,270
        xp = np.rot90(xp, k, (-2, -1)).copy(); yp = np.rot90(yp, k, (-2, -1)).copy()
        return torch.from_numpy(xp), torch.from_numpy(yp)

train_dl = DataLoader(PatchDS(train_ids), batch_size=BATCH, shuffle=True,
                      num_workers=0, pin_memory=(device.type=="cuda"))
val_dl   = DataLoader(PatchDS(val_ids),   batch_size=BATCH, shuffle=False,
                      num_workers=0, pin_memory=(device.type=="cuda"))

# -----------------
# model
# -----------------
class SmallUNet(nn.Module):
    def __init__(self, in_ch, ncls):
        super().__init__()
        def block(c1,c2):
            return nn.Sequential(nn.Conv2d(c1,c2,3,padding=1), nn.ReLU(True),
                                 nn.Conv2d(c2,c2,3,padding=1), nn.ReLU(True))
        self.d1 = block(in_ch, 32); self.p1 = nn.MaxPool2d(2)
        self.d2 = block(32, 64);    self.p2 = nn.MaxPool2d(2)
        self.b  = block(64,128)
        self.u2 = nn.ConvTranspose2d(128,64,2,2)
        self.u1 = nn.ConvTranspose2d(64,32,2,2)
        self.c2 = block(128,64)
        self.c1 = block(64,32)
        self.out= nn.Conv2d(32, ncls, 1)
    def forward(self,x):
        x1=self.d1(x); x2=self.d2(self.p1(x1)); xb=self.b(self.p2(x2))
        x = self.u2(xb); x = torch.cat([x,x2],1); x=self.c2(x)
        x = self.u1(x);  x = torch.cat([x,x1],1); x=self.c1(x)
        return self.out(x)

K = np.load(base/f"data_{train_ids[0]}.npy").shape[-1]
model = SmallUNet(K, NCLASSES).to(device)

# -----------------
# training setup
# -----------------
params = sum(p.numel() for p in model.parameters())
print(f"[model] SmallUNet(in_ch={K}, ncls={NCLASSES}), params={params/1e6:.2f}M")
print(f"[train] device={device}, PATCH={PATCH}, BATCH={BATCH}, EPOCHS={EPOCHS}")

opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
lossfn = nn.CrossEntropyLoss(weight=class_weights)

best_model = {
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
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad()
        logits = model(xb)
        loss = lossfn(logits, yb)
        loss.backward()
        opt.step()
        running += loss.item()
        if b % 100 == 0:
            print(f"  [epoch {epoch}] batch {b}: loss={loss.item():.4f}")

    # validation
    model.eval()
    total_pred, total_gt = [], []
    with torch.no_grad():
        for xb,yb in val_dl:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1).cpu().numpy().ravel()
            gt = yb.cpu().numpy().ravel()
            total_pred.append(pred); total_gt.append(gt)
    pred = np.concatenate(total_pred)
    gt = np.concatenate(total_gt)
    dt = time.time() - t0

    miou, iou_per_class = miou_from_preds_gts(pred, gt, ncls=NCLASSES)
    print(f"Epoch: {epoch:02d}  mIoU: {miou:.3f}  IoU[Cloud, Land, Sea]: {np.round(iou_per_class,3)}  time: {dt:.1f}s")

    if miou > best_model["miou"]:
        best_model["miou"] = miou
        best_model["epoch"] = epoch
        best_model["state"] = copy.deepcopy(model.state_dict())
        best_model["iou_per_class"] = iou_per_class.copy()
        print(f"  [best] new best mIoU {miou:.3f} at epoch {epoch}")

# -----------------
# end-of-run confirm save
# -----------------
print("\nTraining complete.")
print(f"Best mIoU: {best_model['miou']:.3f} at epoch {best_model['epoch']}  IoU/cls: {np.round(best_model['iou_per_class'],3)}")

ans = input("Will you save this model? (yes/no): ").strip().lower()
if ans in {"y", "yes"} and best_model["state"] is not None:
    tag = f"e{best_model['epoch']}_miou{best_model['miou']:.3f}".replace('.', '-')
    ckpt_path = out / f"smallunet_{tag}.pt"
    torch.save({
        "model": "SmallUNet",
        "state_dict": best_model["state"],
        "epoch": best_model["epoch"],
        "miou": best_model["miou"],
        "iou_per_class": best_model["iou_per_class"],
        "config": {
            "PATCH": PATCH, "BATCH": BATCH, "EPOCHS": EPOCHS,
            "NCLASSES": NCLASSES, "train_ids": train_ids, "val_ids": val_ids
        }
    }, ckpt_path)
    print(f"[save] saved model to: {ckpt_path}")
else:
    print("[save] skipped saving model.")
