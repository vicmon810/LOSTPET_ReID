# app/scripts/train_ssl_hf.py
import os
import torch, torchvision as tv
from torch import nn, optim
from datasets import load_dataset
import timm
from pathlib import Path

# --- device: use CPU for now (GTX 1060 sm_61 not supported by newer torch wheels) ---
# device = "cuda" if torch.cuda.is_available() else "cpu"

device = "cpu"

# --- keep workers sane (avoid freeze/slow) ---
num_workers = min((os.cpu_count() or 4) - 2, 6)
num_workers = max(num_workers, 0)

# 1) Load Stanford Dogs from HF (community mirror)
ds = load_dataset("maurice-fp/stanford-dogs")

# 2) We only need PIL images for SSL
def to_pil_only(e): 
    return {"image": e["image"]}
ds = ds.with_transform(to_pil_only)

# 3) Augmentations
tfm = tv.transforms.Compose([
    tv.transforms.Lambda(lambda im: im.convert("RGB")), 
    tv.transforms.RandomResizedCrop(224),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(0.4,0.4,0.4,0.1),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def collate(batch):
    imgs = [tfm(x["image"]) for x in batch]
    return torch.stack(imgs, dim=0)

train_loader = torch.utils.data.DataLoader(
    ds["train"], batch_size=64, shuffle=True, num_workers=num_workers,
    collate_fn=collate, drop_last=True
)

# 4) Build DINOv2 backbone that expects 224x224
backbone = timm.create_model(
    "vit_small_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,
    img_size=224,    # IMPORTANT: match transforms
).to(device)

proj = nn.Sequential(
    nn.Linear(backbone.num_features, 256), nn.ReLU(), nn.Linear(256, 128)
).to(device)

# -- helper: always reduce to [B, C] global features --
def global_feat(feats):
    """
    Convert various timm outputs to a [B, C] tensor.
    Handles dict outputs and tensor outputs robustly.
    """
    if isinstance(feats, dict):
        # Try CLS-like tokens first
        for k in ("x_norm_clstoken", "x_norm_clspool", "cls_token", "pooled_token"):
            v = feats.get(k, None)
            if isinstance(v, torch.Tensor):
                return v.squeeze(1)  # [B,1,C] -> [B,C]
        # Then patch tokens [B,N,C] -> mean over N
        for k in ("x_norm_patchtokens", "tokens"):
            v = feats.get(k, None)
            if isinstance(v, torch.Tensor) and v.dim() == 3:
                return v.mean(dim=1)
        # Fallback to any tensor field we can find
        for k in ("last_norm", "features", "res"):
            v = feats.get(k, None)
            if isinstance(v, torch.Tensor):
                feats = v  # fallthrough
                break
        else:
            raise RuntimeError("Cannot find usable feature in feats dict.")

    # Tensor branch
    if isinstance(feats, torch.Tensor):
        if feats.dim() == 4:         # [B,C,H,W] -> GAP
            return feats.mean(dim=(2,3))
        if feats.dim() == 3:         # [B,N,C] -> mean tokens
            return feats.mean(dim=1)
        if feats.dim() == 2:         # [B,C]
            return feats

    raise RuntimeError(f"Unexpected feature shape/type: {type(feats)} / {getattr(feats,'shape',None)}")

# 5) Simple InfoNCE over batch embeddings [B, D]
def info_nce(z, temp=0.1):
    # z: [B, D]
    z = nn.functional.normalize(z, dim=-1)
    # cosine similarity matrix
    sim = z @ z.t() / temp  # [B,B]
    # Use instance discrimination (diagonal as positives)
    labels = torch.arange(z.size(0), device=z.device)
    return nn.functional.cross_entropy(sim, labels)

opt = optim.AdamW(
    list(backbone.parameters()) + list(proj.parameters()),
    lr=3e-4, weight_decay=1e-4
)

Path("checkpoints").mkdir(parents=True, exist_ok=True)

epochs = 10
for ep in range(epochs):
    backbone.train(); proj.train()
    for imgs in train_loader:
        imgs = imgs.to(device)
        feats_raw = backbone.forward_features(imgs)
        feats_vec = global_feat(feats_raw)  # -> [B, C]
        z = proj(feats_vec)                 # -> [B, 128]
        # guard small batches (shouldn't happen with drop_last=True, but just in case)
        if z.size(0) < 2:
            continue
        loss = info_nce(z)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[epoch {ep}] ssl loss={loss.item():.4f}")

torch.save({"backbone": backbone.state_dict()}, "checkpoints/dog_ssl_backbone_hf.pt")
print("Saved: checkpoints/dog_ssl_backbone_hf.pt")
