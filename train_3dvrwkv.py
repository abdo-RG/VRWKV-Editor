import torch
import diffusers
import transformers
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils import check_min_version
from diffusers import VQModel, AutoencoderKL, DDPMScheduler, DDIMScheduler
from diffusers.optimization import get_scheduler
from accelerate.logging import get_logger

import decord
decord.bridge.set_bridge('torch')
from einops import rearrange
from tqdm.auto import tqdm
import os
import argparse
import inspect
import math
from typing import Dict, Optional, Tuple
from omegaconf import OmegaConf
import logging

import torch.nn.functional as F
from torch.utils.data import Dataset
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.utils import set_seed

from models.unet3d import SwinUnetModel
from models.Transf_Blocks.transformers3d import configure_channel_mix, configure_spatial_mix, configure_arch, configure_temporal
from util import save_videos_as_images, SwinPipe, inversion, save_videos



# Prepare data for training and inference
class SwinDataset(Dataset):
    def __init__(
            self,
            video_path: str,
            prompt: str,
            frame_rate: int = 2,
            n_sample_frames: int = 16,     
    ):
        self.width = self.height = 512
        self.frame_rate = frame_rate
        self.n_sample_frames = n_sample_frames
        self.video_path = video_path
        self.prompt = prompt
        self.prompt_ids = None


    def __len__(self):
        return 1

    def __getitem__(self, index):
        # load and sample video frames
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(0, len(vr), self.frame_rate))[:self.n_sample_frames]
        video = vr.get_batch(sample_index)
        video = rearrange(video, "f h w c -> f c h w")

        example = {
            "pixel_values": (video / 127.5 - 1.0),
            "prompt_ids": self.prompt_ids
        }

        return example


logger = get_logger(__name__, log_level="INFO")


def main(
    train_data: Dict,
    validation_data: Dict,
    output_dir: str,
    updated_modules: Tuple[str],
    validation_steps: int = 100,
    max_train_steps: int = 500,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    use_channel_mix: bool = False,
    channelmix_act: str = "sq_relu",
    base_lr: float = 3e-5,
    mu_lr_mult: float = 1.0,
    shift_pixel: int = 1,
    channel_gamma: float = 0.25,
    attention_heads: Optional[int] = None,
    spatial_mode: str = "both",
    temporal_mixer: Optional[str] = None,):
    
    # Fix the randomness to produce the same sequence of random numbers
    set_seed(33)
    *_, config = inspect.getargvalues(inspect.currentframe())
    

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision="fp16",
    )
    
    # Initializing scheduler, tokenizer and models.
    pretrained_model_path = "checkpoints/stable-diffusion-v1-4"
    noise_scheduler = DDPMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Register the output state for each samlpe
    if accelerator.is_main_process:
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/samples", exist_ok=True)
        os.makedirs(f"{output_dir}/inv_latents", exist_ok=True)
        OmegaConf.save(config, os.path.join(output_dir, 'config.yaml'))
    
    
    # Enable optional VRWKV channel-mixing (R1.3 activation ablation) before building the U-Net.
    configure_channel_mix(use_channel_mix, channelmix_act)
    if use_channel_mix:
        logger.info(f"Channel-mixing ENABLED with activation='{channelmix_act}'")

    # Architecture mode: 'both' (orig) | 'linear_only' (drop quadratic attn1) | 'quad_only'.
    configure_arch(spatial_mode)
    # R1.4 controlled temporal-mechanism benchmark (None -> deployed VRWKV temporal).
    configure_temporal(temporal_mixer)
    logger.info(f"[arch] spatial_mode={spatial_mode} temporal_mixer={temporal_mixer or 'vrwkv(default)'}")

    # R1.6 sensitivity hooks: spatial token-shift hyperparameters + attention head count.
    configure_spatial_mix(shift_pixel, channel_gamma)
    if attention_heads is not None:
        os.environ["VRWKV_ATTENTION_HEADS"] = str(int(attention_heads))
    logger.info(f"[R1.6] shift_pixel={shift_pixel}, channel_gamma={channel_gamma}, "
                f"attention_heads={attention_heads if attention_heads is not None else 'default(8)'}")

    unet = SwinUnetModel.from_pretrained_2d(pretrained_model_path, subfolder="unet")
    unet.requires_grad_(False)
    for name, module in unet.named_modules():
        if name.endswith(tuple(updated_modules)):
            for params in module.parameters():
                params.requires_grad = True

    unet.enable_xformers_memory_efficient_attention()
    unet.enable_gradient_checkpointing()

    # Initialize the optimizer
    optimizer_cls = torch.optim.AdamW

    # Optional separate (higher-LR, no-weight-decay) param group for the temporal
    # interpolation / scale parameters (mu, mu_c, gamma_t, gamma_c, time_decay,
    # time_first). This is the R1-comment-3 diagnostic: give these parameters a
    # fair chance to move, judged by whether they (a) develop channel-wise
    # variation and (b) change output metrics. mu_lr_mult=1.0 -> standard recipe.
    MU_SUFFIXES = ("attn_temp.mu_logit", "attn_temp.mu_c_logit", "gamma_t", "gamma_c",
                   "time_decay", "time_first")
    mu_params, base_params = [], []
    for name, p in unet.named_parameters():
        if not p.requires_grad:
            continue
        (mu_params if name.endswith(MU_SUFFIXES) else base_params).append(p)
    param_groups = [{"params": base_params, "lr": base_lr, "weight_decay": 1e-2}]
    if mu_lr_mult != 1.0 and mu_params:
        param_groups.append({"params": mu_params, "lr": base_lr * mu_lr_mult, "weight_decay": 0.0})
        print(f"[mu-diagnostic] {len(mu_params)} temporal interp/scale tensors at "
              f"lr={base_lr*mu_lr_mult:g} (x{mu_lr_mult}); {len(base_params)} others at lr={base_lr:g}")
    else:
        param_groups = [{"params": base_params + mu_params, "lr": base_lr, "weight_decay": 1e-2}]

    optimizer = optimizer_cls(param_groups, betas=(0.9, 0.999), eps=1e-08,)

    # Get the training dataset
    train_dataset = SwinDataset(**train_data)

    # Preprocessing the dataset
    train_dataset.prompt_ids = tokenizer(
        train_dataset.prompt, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    ).input_ids[0]

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1
    )


    ddim_inv_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder='scheduler')
    ddim_inv_scheduler.set_timesteps(validation_data.num_inv_steps)

    # Scheduler
    lr_scheduler = get_scheduler(
        "constant",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Managing our training everything with our `accelerator`.
    unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        unet, optimizer, train_dataloader, lr_scheduler
    )

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        
    # Instantiate the validation pipeline
    validation_pipeline = SwinPipe(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, unet=unet,
        scheduler=DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    )
    validation_pipeline.enable_vae_slicing()

    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)
    total_batch_size = 1 * accelerator.num_processes * gradient_accumulation_steps
    global_step = 0
    first_epoch = 0
    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("text2video-fine-tune")
    
    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1]
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))
        global_step = int(path.split("-")[1])

        first_epoch = global_step // num_update_steps_per_epoch
        resume_step = global_step % num_update_steps_per_epoch

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(unet):
                # Convert videos to latent space
                pixel_values = batch["pixel_values"].to(weight_dtype)
                video_length = pixel_values.shape[1]
                pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
                latents = latents * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each video
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                encoder_hidden_states = text_encoder(batch["prompt_ids"])[0]

                # Get the target for loss depending on the prediction type
                if noise_scheduler.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

                # Predict the noise residual and compute loss
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(1)).mean()
                train_loss += avg_loss.item() / gradient_accumulation_steps

                # Backpropagate
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    if accelerator.is_main_process:
                        samples = []
                        generator = torch.Generator(device=latents.device)
                        generator.manual_seed(33)

                        ddim_inv_latent = None
                        if validation_data.use_inv_latent:
                            inv_latents_path = os.path.join(output_dir, f"inv_latents/ddim_latent-{global_step}.pt")
                            ddim_inv_latent = inversion(
                                validation_pipeline, ddim_inv_scheduler, video_latent=latents,
                                num_inv_steps=validation_data.num_inv_steps, prompt="")[-1].to(weight_dtype)
                            torch.save(ddim_inv_latent, inv_latents_path)
                        print(f"ddim_inv_latent : {ddim_inv_latent.shape}")

                        for idx, prompt in enumerate(validation_data.prompts):
                            sample = validation_pipeline(prompt, generator=generator, latents=ddim_inv_latent,
                                                         **validation_data).videos
                            save_videos(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif")
                            save_videos_as_images(sample, f"{output_dir}/samples/sample-{global_step}/{prompt}.gif",prompt)
                            samples.append(sample)
                        samples = torch.concat(samples)
                        save_path = f"{output_dir}/samples/sample-{global_step}.gif"
                        save_videos(samples, save_path)
                        logger.info(f"Saved samples to {save_path}")

            logs = {"step_loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        pipeline = SwinPipe.from_pretrained(
            pretrained_model_path,
            text_encoder=text_encoder,
            vae=vae,
            unet=unet,
        )
        pipeline.save_pretrained(output_dir)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/swin.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
