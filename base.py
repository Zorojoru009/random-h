# For Kaggle usage, install dependencies first (uncomment the line below):
# !pip install -U diffusers transformers accelerate safetensors huggingface_hub

import torch
import warnings
# Suppress diffusers/transformers warnings for cleaner agent output
warnings.filterwarnings("ignore")

from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

# Constants for easy configuration
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
LIGHTNING_REPO = "ByteDance/SDXL-Lightning"
LIGHTNING_CKPT = "sdxl_lightning_8step_unet.safetensors" # 8-step UNet for MAXIMUM QUALITY

def load_pipeline(device="cuda"):
    """
    Loads the SDXL Lightning pipeline and returns it.
    Useful for preloading the model on a specific GPU.
    Args:
        device: Device to load the model on (e.g., "cuda:0", "cuda:1").
    """
    # Load UNet checkpoint (better quality than LoRA)
    print(f"Loading SDXL Lightning 8-step UNet on {device}...")
    unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device, torch.float16)
    unet.load_state_dict(load_file(hf_hub_download(LIGHTNING_REPO, LIGHTNING_CKPT), device=device))
            
    # Load Pipeline with custom UNet
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_MODEL, 
        unet=unet,
        torch_dtype=torch.float16, 
        variant="fp16"
    ).to(device)

    # Configure Scheduler for Lightning
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    return pipe

def generate_baroque_image(prompts, negative_prompt="", output_prefix="output", save_to_disk=True, pipe=None):
    """
    Generates Baroque-style images using SDXL Lightning (MAX QUALITY).
    Using 8-step UNet checkpoint with guidance_scale=0 for best quality.
    Locked to 16:9 aspect ratio (1024x576).
    Args:
        prompts: A single string or a list of strings.
        negative_prompt: Universal negative prompt.
        output_prefix: Prefix for saved filenames.
        save_to_disk: Whether to save images to disk (default True).
        pipe: Optional pre-loaded pipeline. If None, loads a new one.
    Returns:
        List of PIL Image objects.
    """
    
    if isinstance(prompts, str):
        prompts = [prompts]

    if pipe is None:
        pipe = load_pipeline()

    print(f"Generating {len(prompts)} images...")
    
    generated_images = []
    for i, prompt in enumerate(prompts):
        # Generation Settings (MAX QUALITY)
        # Official SDXL-Lightning docs: use guidance_scale=0 for trained models
        # 8 steps for highest quality
        # Resolution locked to 1024x576 (16:9)
        image = pipe(
            prompt, 
            negative_prompt=negative_prompt,
            num_inference_steps=8, 
            guidance_scale=0,
            height=576,
            width=1024
        ).images[0]

        generated_images.append(image)
        
        if save_to_disk:
            filename = f"{output_prefix}_{i}.png"
            image.save(filename)
            print(f"[{i+1}/{len(prompts)}] Saved {filename}")
        else:
            print(f"[{i+1}/{len(prompts)}] Image generated")
    
    return generated_images

if __name__ == "__main__":
    # Example usage: Batch generation
    batch_prompts = [
        "baroque oil painting, dramatic chiaroscuro lighting, a massive disembodied eye made of fire, non-human, symbolic",
        "baroque oil painting, dramatic chiaroscuro lighting, a golden crown floating in a dark void, divine light",
        "baroque oil painting, dramatic chiaroscuro lighting, ancient stone statues emerging from shadows",
        # TEXT CAPABILITY TEST:
        "baroque oil painting, dramatic chiaroscuro lighting, close up of a weathered stone tablet with the word 'VERITAS' deeply chiseled into it, sharp typography, cinematic lighting, old master style"
    ]
    
    generate_baroque_image(batch_prompts)
