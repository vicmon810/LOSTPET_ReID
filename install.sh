#install

python -m venv .venv

source .venv/bin/active # Windows: .venv/Scripts/activate

pip install --upgrade pip
pip install fastapi uvicorn timm pillow torchvision faiss-cpu open_clip_torch transformers accelerate-U tensorflow-datasets
