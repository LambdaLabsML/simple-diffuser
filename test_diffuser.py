import torch
from diffusers import StableDiffusionPipeline, UNet2DConditionModel

model_path = "/home/ubuntu/arc-utah/chuan/simple-diffuser/sd-naruto-model/checkpoint-100"
unet = UNet2DConditionModel.from_pretrained(model_path, torch_dtype=torch.float16)

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", unet=unet, torch_dtype=torch.float16)
pipe.to("cuda")



image = pipe(prompt="Yoda doing ninja moves.").images[0]
image.save("yoda-ninja.png")