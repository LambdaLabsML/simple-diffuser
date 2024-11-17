import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "./sd-naruto-model/checkpoint-7500"
unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    unet=unet,
    torch_dtype=torch.float16,
    safety_checker = None,
    requires_safety_checker = False)
pipe.to("cuda")



image = pipe(prompt="Yoda").images[0]
image.save("yoda.png")
