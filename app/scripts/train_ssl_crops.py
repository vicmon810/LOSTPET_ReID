import torch, torchvision as tv
from torch import nn, optim
from pathlib import Path
import timm

root = Path("data/stanford_dogs_crops")
device = "cuda" if torch.cuda.is_available() else "cpu"

tfm = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ColorJitter(0.4,0.4,0.4,0.1),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

ds = tv.datasets.ImageFolder(str(root), transform=tfm)
dl = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=True, num_workers=8, drop_last=True)

backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0).to(device)
proj = nn.Sequential(nn.Linear(backbone.num_features, 256), nn.ReLU(), nn.Linear(256, 128)).to(device)

def info_nce(z, temp=0.1):
    z = nn.functional.normalize(z, dim=-1)
    sim = z @ z.t() / temp
    labels = torch.arange(z.size(0), device=z.device)
    return nn.functional.cross_entropy(sim, labels)

opt = optim.AdamW(list(backbone.parameters())+list(proj.parameters()), lr=3e-4, weight_decay=1e-4)
Path("checkpoints").mkdir(parents=True, exist_ok=True)

for ep in range(5):
    backbone.train(); proj.train()
    for x, _ in dl:
        x = x.to(device)
        feats = backbone.forward_features(x)
        if isinstance(feats, dict):
            tokens = feats.get("x_norm_patchtokens", None)
            if tokens is not None:
                feats = tokens.mean(dim=1)
            else:
                feats = feats if isinstance(feats, torch.Tensor) else feats.get("x_norm_clstoken").squeeze(1)
        z = proj(feats)
        loss = info_nce(z)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[epoch {ep}] ssl loss={loss.item():.4f}")

torch.save({"backbone": backbone.state_dict()}, "checkpoints/dog_ssl_backbone_crops.pt")
print("Saved: checkpoints/dog_ssl_backbone_crops.pt")
