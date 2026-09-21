# VRWKV-Editor

**Reducing Quadratic Complexity in Transformer-Based Video Editing.**

VRWKV-Editor is a one-shot, text-driven video editing framework that replaces the
quadratic spatio-temporal attention of diffusion-based editors with a **linear**
`VRWKV`-style aggregation module (bidirectional Weighted Key-Value, Bi-WKV). It is
built on top of a Tune-A-Video-style pipeline with a Stable Diffusion v1.4
backbone, and reduces the memory and runtime of the spatial-aggregation module by
up to **~72% vs. Swin attention** and **~96% vs. sparse-causal attention**, with
the gap widening at longer sequences and higher resolutions, while preserving
temporal consistency and text alignment.

Paper: *VRWKV-Editor: Reducing Quadratic Complexity in Transformer-Based Video
Editing* project page: https://abdo-rg.github.io/VRWKV-Editor/

---

## Method overview

Given a single text–video pair, the model is briefly tuned on that clip and can
then re-synthesize it under an edited prompt (object replacement, background
change, style transfer) while preserving the source motion.

The core is the **3D-VRWKV module**, an alternative for spatio-temporal
attention with two linear-complexity sub-modules:

1. **Spatio-temporal mixing:** quad-directional token shift + a learnable
   channel-wise temporal interpolation (μ) followed by bidirectional Bi-WKV
   aggregation.
2. **Channel mixing:** a second learnable interpolation (μ_c) plus a squared-ReLU
   gate for cross-channel communication.

Both μ and μ_c are learned per channel and adaptively control temporal smoothness.

```
repo/
├── train_3dvrwkv.py     # tune the model on one video (entry point)
├── test.py                 # edit a video with a trained checkpoint
├── util.py                 # SwinPipe pipeline, DDIM inversion, video I/O
├── models/                 # U-Net + 3D-VRWKV modules
│   ├── unet3d.py           #   SwinUnetModel (from_pretrained_2d inflation)
│   ├── unet_blocks.py
│   ├── Transf_Blocks/      #   attentions.py (Bi-WKV, controlled mixers),
│   │                       #   transformers3d.py (block + config hooks)
│   └── Res_Blocks/
├── configs/                # per-video training/editing configs (YAML)
├── data/                   # example input video (man-skiing.mp4)
```

---

## Installation

```bash
conda create -n 3vdrwkv python=3.10 -y
conda activate 3dvrwkv
pip install -r requirements.txt
```

The pinned versions (torch 2.2.0, diffusers 0.11.1, xformers 0.0.24, etc.) are the
ones the paper was run with. A CUDA GPU is required (experiments used an
A100-80GB); `xformers` is used for memory-efficient attention.

### Pretrained backbone

Download the Stable Diffusion v1.4 weights and place them where the configs expect
them (default `checkpoints/stable-diffusion-v1-4`):

```bash
git lfs install
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 \
    checkpoints/stable-diffusion-v1-4
```

---

## Usage

### 1. Tune on a single video

Each YAML in `configs/` describes one video: its path, source prompt, the target
edit prompts, and the training budget.

```bash
python train_3dvrwkv.py --config configs/man-skiing.yaml
```

This writes the tuned U-Net, the DDIM inversion latents, and validation samples to
the config's `output_dir` (e.g. `outputs/man-skiing/`).

Minimal config schema:

```yaml
output_dir: outputs/man-skiing
train_data:
  video_path: data/man-skiing.mp4
  prompt: a man is skiing          # source description
  n_sample_frames: 16
validation_data:
  prompts:                         # edit prompts sampled during training
    - spider man is skiing on the beach, cartoon style
    - wonder woman, wearing a cowboy hat, is skiing
  video_length: 16
  num_inference_steps: 50
  guidance_scale: 12.5
  use_inv_latent: true
  num_inv_steps: 50
max_train_steps: 500
updated_modules: ["attn1", "attn2.to_q", "attn_temp", "swa_attn1"]
```

Optional keys (used for the ablations in the paper — omit for the default model):
`spatial_mode` (`both`/`linear_only`/`quad_only`), `temporal_mixer`
(`full_attn`/`wkv`/`linear_attn`/`ssm`), `use_channel_mix`, `channelmix_act`,
`shift_pixel`, `channel_gamma`, `attention_heads`.

### 2. Edit a video with a tuned checkpoint

```bash
python test.py
```

`test.py` loads a tuned checkpoint and its inversion latent, then generates an
edited clip. The key lines to adapt for your own run:

```python
my_model_path = "./outputs/man-skiing"
unet = SwinUnetModel.from_pretrained(my_model_path, subfolder="unet",
                                     torch_dtype=torch.float16).to("cuda")
pipe = SwinPipe.from_pretrained("checkpoints/stable-diffusion-v1-4",
                                unet=unet, torch_dtype=torch.float16).to("cuda")

prompt = "spider man is skiing"
inv_latent = torch.load(f"{my_model_path}/inv_latents/ddim_latent-500.pt").half()
video = pipe(prompt, latents=inv_latent, video_length=16, height=512, width=512,
             num_inference_steps=50, guidance_scale=12.5).videos
save_videos(video, f"./{prompt}.gif")
```

## Citation

```bibtex
@article{aitrouga2026vrwkv,
  title={VRWKV-Editor: reducing quadratic complexity in transformer-based video editing},
  author={Aitrouga, Abdelilah and Hmamouche, Youssef and Seghrouchni, Amal El Fallah},
  journal={Multimedia Systems},
  volume={32},
  number={9},
  pages={586},
  year={2026},
  doi = {https://doi.org/10.1007/s00530-026-02649-4},
  publisher={Springer}
}
```

## Acknowledgements

Built on [Tune-A-Video](https://github.com/showlab/Tune-A-Video),
[Diffusers](https://github.com/huggingface/diffusers), and the
[Vision-RWKV](https://github.com/OpenGVLab/Vision-RWKV) formulation. Thanks for open-sourcing!
