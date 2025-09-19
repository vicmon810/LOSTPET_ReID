# LOSTPET-REID 

This project provides a **dog re-identification (Re-ID) system** powered by **FastAPI**, **DINOv2**, **CLIP**, and **FAISS**.  
It is designed for pet search/retrieval, where users can register a pet’s photo and later query with a new photo (optionally guided by text prompts like "red collar").

---

##  Project Structure

```
. (Root of this repository)
└── app/
    ├── main.py #FASTAPI wiht register/search endpoint
    ├── images
    |       └── stanford_dog
    |       └── more...
    |       └── more
    ├── script
            ├── crop_dogs.py # preprocessing
            ├── train.py # self-supervised 
            └── TBC
         
```

## Self-Supervised training 

### Architecture 
    - overall framework 
        - paradigm: self-supervied learning(SSL) warm-up -> fearue extraction retrieval 
        - currently running on self-supervised pretraining stage so the backbone learns general dog texture, shapes and body parts 
        - after training, the backbone is used insdie the retrieval service, paried with CLIP heatmap(for attention guidance) and FAISS(for similarity ) 
    - Backbone: DINOv2 ViT-S/14
        - Model `***vit_small_patch14_dinov2.lvd142m***` from timm[https://huggingface.co/timm]
        - type: Vision Transformer(small, patch size 14)
        - pretraining: Comes with ***DINOv2 self-supervised weights.***
        - Input resolution: Forced to 224×224 for compatibility and efficiency (the original checkpoint was trained at 518×518).
        - Output: Global embedding vector (~384 dim).
### Projection Head 
- Small MLP 

```
Linear(384 -> 256) -> ReLU -> linear(256 -> 128)
```

- purpose: reduce the feature dimension to 128 for InfoNCE loss.

### Loss: Simplified InfoNCE

- Current simple variant(toy)
- steps: 
    - Normalize embeddings (z = normalize(z))
    - compute similarity matrix 
    - use diagonal as positivies and other as negatives

### Training Loop
- optimizer: Adamw(lr=3e-4, weight decay=1e-4)
- Augmentations: 
    - RandomResizedCrop(244)
    - RandomHorizontalFlip
    - ColorJitter
    - ToTensor + normalization (mean=[0.5]*3, std=[0.5]*3)
- Batch size: 64
- Workers: Limited to 6 (avoid DataLoader warnings).

## Source
-  Hugging Face community dataset: stanford-dogs[https://huggingface.co/datasets/maurice-fp/stanford-dogs]
- This is a mirror of Stanford Dogs (120 breeds, ~20k images).

- Contains images + labels, but labels are ignored for SSL training.

- Preprocessing

    - Convert all images to RGB (fixes RGBA errors).

    - Apply augmentations for contrastive/self-supervised learning.

## Setup

```
# quick start  

# install  dependencies 

bash install.sh

```