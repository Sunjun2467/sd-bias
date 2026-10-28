import os
os.environ["HF_HOME"] = "models/hf_cache"

from huggingface_hub import snapshot_download

snapshot_download("stable-diffusion-v1-5/stable-diffusion-v1-5",
                  local_dir="models/sd15",
                  allow_patterns=["*.json", "*.txt", "*fp16.safetensors"])

snapshot_download("openai/clip-vit-large-patch14",
                  local_dir="models/clip-l14",
                  allow_patterns=["*.json", "*.txt", "*.safetensors"])

print("done")