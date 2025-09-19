# app/main.py

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel 
import io, torch, faiss
from PIL import Image 
import numpy as np 
import torchvision.transforms as T
import timm
import open_clip 

app = FastAPI(title = "LostPet-ReID") 

# Model & index 
device = "cuda" if torch.cuda.is_available() else "cpu"

# DINOv2 for visual embedding 
dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=False)
dino.reset_classifier(num_classes=0) 
# dino.eval().to(device) 
ckpt_path_candiates = []

ckpt_loaded = False

for ckpt_path in ckpt_path_candiates:
    if os.path.isfile(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        # {"backbone" : state_dict} or state_dict 
        state_dict = ckpt["backbone"] if isinstance(ckpt, dict) and "backbone" in ckpt else ckpt 

        fixed_state = {}

        for k,v,in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k 
            fixed_state[nk] = v 

        missing, unexpected = dino.load_state_dict(fixed_state, strict = False)
        print(f"[DINO] loaded SSL backbone from: {ckpt_path}")
        print(f"[DINO] missing key: {len(missing)} unexpect keys {len(unexpect)}")
        ckpt_loaded = True
        break 
# defalue offical dino
if not ckpt_loaded:
    print("[DINO] No custom SSL checkpoint found, falling back to pretrained=True")
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True)
    dino.reset_classifier(num_classes=0)

dino.eval().to(device)

# 4) （可选）快速自检：前向一次看维度是否正常
with torch.no_grad():
    dummy = torch.zeros(1, 3, 224, 224, device=device)
    feats = dino.forward_features(dummy)
    if isinstance(feats, dict):
        # 绝大多数 timm ViT 会给出 token（N 个 patch）与 cls token
        tok = feats.get("x_norm_patchtokens", None)
        cls = feats.get("x_norm_clstoken", None)
        if tok is not None:
            print(f"[DINO] patch tokens: {tok.shape}")  # (1, N, C)
        if cls is not None:
            print(f"[DINO] cls token: {cls.shape}")      # (1, 1, C)
    else:
        # 有些版本直接返回 tensor
        print(f"[DINO] features tensor: {feats.shape}")

#CLIP for text-guided heatmap 
clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
)

clip_model.eval().to(device)
clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

#FAISS index (cosine similarity \approx inner product when vector are L2-norm)
DIM = dino.num_features
index = faiss.IndexHNSWFlat(DIM,32,faiss.METRIC_INNER_PRODUCT) 

# In-memory gallery metadat
gallery = [] # e.g., {"pet_id":str, "image_key":str, "vector":np.ndarray}

#Preprocessing 
to_tensor = T.Compose([
    T.Resize(224, interpolation=T.InterpolationMode.BICUBIC),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(mean=0.5, std=0.5)
])

def l2n(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) +1e-6)

@torch.inference_mode()
def dino_embed(img:Image.Image, heatmap:np.ndarray|None):
    """
    Return a normalized embedding 
    If heatmap exists, use it to weight Vit path tockens
    """
    x = to_tensor(img).unsqueeze(0).to(device) #(1,3,244,244) 

    feat = dino.forwar_features(x)
    tokens = None
    cls = None

    if isinstance(feat, dict):
        # Common key in recent timm for Vit:
        tokens = feats.get("x_norm_patchtokens", None) #(1, N, C) 
        cls = feats.get("x_norm_clstoken", None) #(1,1,C) 

        if tokens is None: 
            # Fallback some variants expose 'token' differently
            pass
    else:#some model just return a tensor, fallback
        pass

    if tokens is not None and heatmap is not None: 
        # Map heatmap HxW -> 16 * 16(224/14) 
        H = torch.from_numpy(heatmap).to(device)
        H = (H - H.min() / (H.max() - H.min() + 1e-6 ))
        H = torch.nn.functional.interpolate(
                H[None, None, ...], size = (16,16), mode="bilinear", align_corners=False).view(1,-1) #(1,256)
        w = H / (H.sum(dim=-1, keepdim=True) + 1e-6) #(1,256) weight sum to 1
        e = (token * w.unsqueeze(-1)).sum(dim=1)  # (1,C)
    else: #No token or no heatmap: average tokens or fallback 
        if tokens is not None: e = tokens.mean(dim=1) # (1,C)
        elif isinstance(feats, torch.Tensor): e = feats.mean(dim=(-2,-1), keepdim=False) #global pool is spatial 
        elif cls is not None: 
            e = cls.squeeze(1)
        else:
            raise RuntimeError("Cannot obtain embedding from DINO features.") 
    e = ln2(e).squeeze(0) # (C,) 
    return e 


@torch.inference_mode()
def clip_text_heatmap(img:Image.Image, text:str|None):
    """
    return (heatmap 224*224)
    """
    if not text or text.strip():
        return None, None

    clip_img = clip_preprocess(img).unsqueeze(0).to(device) 
    tok = clip_tokenizer([text]).to(device) 

    #Compute similartity 
    img_feature = clip_model.encode_image(clip_img) 
    text_feature = clip_model.encode_text(tok)
    image_feature = img_feature / img_feature.norm(dim=-1, keepdim=True) 
    text_feature = text_feature / text_feature.norm(dim=-1, keepdim=True) 
    score = (image_feature @ text_feature.T).item() 

    # Placehoder heatmap 
    g = np.array(img.convert("L").resize((224,224))) /255.0
    heat = (g - g.min() ) /(g.max() - g.min() + 1e-6) 
    return heat.astype(np.float32), score 

# API Class 
      
class RegisterResp(BaseModel):
    count:int 

class MatchItem(BaseModel): 
    pet_id: str
    image_key: str
    similarity: float

#API endpoint

@app.post("/register_pet", response_model = RegisterResp)
async def register_pet(pet_id:str = Form(...), file:UploadFile =None):
        raw = await file.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        e = dino_embed(img, heatmap=None).detach().cpu().numpy().astype("float32")
        # add to FAISS 
        index.add(e[None,:])
        gallery.append({"pet_id":pet_id, "image_key":file.filename, "vector":e})
        return ReisterResp(count=len(gallery) )


@app.post("/search", response_model=SearchResp)
async def search(file: UploadFile, hint:str = Form(None), topk:int = Form(5)):
    raw = await file.read()
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    heat, clip_score = clip_text_heatmap(img, hint) 
    e = dino_embed(img, heatmap=heat).detach().cpu().numpy().astype("float32") 
    D,I = index.search(e[None,:],topk)

    res = []
    
    for d, i in zip(D[0].tolist(), I[0].tolist()):
        if i < 0 or i >= len(gallery):
            continue 
        meta = gallery[i]
        res.append({
            "pet_id": meta["pet_id"]
            ,"image_key":meta["image_key"]
            ,"similarity":float(d)
            })

        return JSONResponse({
            "matches":res, 
            "clip_score":clip_score, 
            "heatmap":heat.tolist() if heat is not None else None
            })


