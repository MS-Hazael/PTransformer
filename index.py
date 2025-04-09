# Instala primero:
# pip install torch diffusers transformers accelerate

from diffusers import StableDiffusionPipeline
import torch

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float32  # Usa float32 en CPU
)

image = pipe("carro deportivo").images[0]
image.save("carro.png")