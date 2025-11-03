from pathlib import Path
import torch, torch.nn as nn
from torchviz import make_dot

# --- your model definition (SmallUNet) here ---
class SmallUNet(nn.Module):
    def __init__(self, in_ch, ncls):
        super().__init__()
        def block(c1,c2):
            return nn.Sequential(
                nn.Conv2d(c1,c2,3,padding=1), nn.ReLU(True),
                nn.Conv2d(c2,c2,3,padding=1), nn.ReLU(True)
            )
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

# === config ===
K = 16          # input channels (e.g., PCA components)
PATCH = 64      # H=W of your patch
NCLASSES = 3

# Optional: path to your saved best model
ckpt_path = Path("model/h2_pca16/smallunet_e2_miou0-767.pt")

# 1) build model
model = SmallUNet(in_ch=K, ncls=NCLASSES).eval()

# 2) (optional) load weights
if ckpt_path.exists():
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

# 3) dummy input with your real input shape
dummy = torch.randn(1, K, PATCH, PATCH)

# 4) forward once to get output
logits = model(dummy)  # shape [1, NCLASSES, PATCH, PATCH]

# 5) build a graph
dot = make_dot(
    logits,
    params=dict(list(model.named_parameters())),
    show_attrs=False, show_saved=False
)

# 6) save only the DOT source (no rendering)
out_base = ckpt_path.with_suffix("") if ckpt_path.exists() else Path("smallunet_graph")
dot_path = f"{out_base}.dot"
dot.save(dot_path)

print(f"[ok] Saved Graphviz DOT file: {dot_path}")
print("You can visualize it here:")
print("  https://dreampuf.github.io/GraphvizOnline")
print("or here:")
print("  https://edotor.net/")
