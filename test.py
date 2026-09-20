from models.unet3d import SwinUnetModel
from util import save_videos_as_images, SwinPipe, inversion, save_videos
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
import torch

validation_data = {
    "prompts": [
        "mickey mouse is skiing on the snow",
        "spider man is skiing on the beach, cartoon style",
        "wonder woman, wearing a cowboy hat, is skiing",
        "a man, wearing pink clothes, is skiing at sunset"
    ],
    "video_length": 16,
    "frame_rate": 2,
    "num_inference_steps": 50,
    "guidance_scale": 12.5,
    "use_inv_latent": True,
    "num_inv_steps": 50
}

# Initializing scheduler, tokenizer and models.
pretrained_model_path = "checkpoints/stable-diffusion-v1-4"
"""noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")"""

my_model_path = "./outputs/man-skiing"
unet = SwinUnetModel.from_pretrained(my_model_path, subfolder='unet', torch_dtype=torch.float16).to('cuda')

# Instantiate the validation pipeline
validation_pipeline = SwinPipe.from_pretrained(pretrained_model_path, unet=unet, torch_dtype=torch.float16).to('cuda')
validation_pipeline.enable_xformers_memory_efficient_attention()
validation_pipeline.enable_vae_slicing()

prompt = "spider man is skiing"
ddim_inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").to(torch.float16)
video = validation_pipeline(prompt, latents=ddim_inv_latent, video_length=16, height=512, width=512, num_inference_steps=50, guidance_scale=12.5).videos
save_videos(video, f"./{prompt}.gif")
