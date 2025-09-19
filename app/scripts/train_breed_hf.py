import torch, torchvision as tv
from torch import nn, optim
from datasets import load_dataset
import timm
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
ds = load_dataset("stanford_dogs")

# 映射为 (tensor, label)
tfm = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.5]*3, std=[0.5]*3),
])

def transform_batch(batch):
    images = [tfm(img) for img in batch["image"]]
    return {"pixel_values": torch.stack(images), "labels": torch.tensor(batch["label"])}

train = ds["train"].with_transform(transform_batch)
test  = ds["test"].with_transform(transform_batch)

train_loader = torch.utils.data.DataLoader(train, batch_size=64, shuffle=True, num_workers=8)
test_loader  = torch.utils.data.DataLoader(test,  batch_size=64, shuffle=False, num_workers=8)

model = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=120).to(device)
opt = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
ce = nn.CrossEntropyLoss()

for epoch in range(5):
    model.train()
    for batch in train_loader:
        x = batch["pixel_values"].to(device); y = batch["labels"].to(device)
        logits = model(x)
        loss = ce(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"[epoch {epoch}] train loss={loss.item():.4f}")

torch.save(model.state_dict(), "checkpoints/dog_breed_classifier_hf.pt")
print("Saved: checkpoints/dog_breed_classifier_hf.pt")
