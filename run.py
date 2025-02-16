from diffusers import StableDiffusionPipeline
from icecream import ic

model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id)

# 2. Zbadaj strukturę UNeta
unet = pipe.unet
ic(unet)  # Zobacz architekturę

# 3. Zbadaj strukturę DiffusionPipeline
# ic(pipe)  # Zobacz strukturę DiffusionPipeline\