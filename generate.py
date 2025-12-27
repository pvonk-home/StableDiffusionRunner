# %%
import argparse
import glob
import os
from datetime import datetime
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from IPython.display import display

# -----------------------------
# Helper: Find most recent image
# -----------------------------
def get_latest_image(output_dir="outputs"):
    images = sorted(
        glob.glob(os.path.join(output_dir, "*.png")),
        key=os.path.getmtime,
        reverse=True
    )
    return images[0] if images else None


# -----------------------------
# Main generation function
# -----------------------------
def generate_image(prompt, init_image_path=None, strength=0.3, width=512, height=512):
    os.makedirs("outputs", exist_ok=True)

    # Load base text-to-image pipeline
    txt2img = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    # If an init image is provided, switch to img2img
    if init_image_path:
        print(f"Using init image: {init_image_path}")
        img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch.float16,
            safety_checker=None
        ).to("cuda")

        init_image = Image.open(init_image_path).convert("RGB")
        init_image = init_image.resize((512, 512))

        result = img2img(
            prompt=prompt,
            image=init_image,
            strength=strength,
            width=width,
            height=height
        )
        image = result.images[0]

    else:
        print("Generating from scratch...")
        result = txt2img(prompt)
        image = result.images[0]

    # Save with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"outputs/output_{timestamp}.png"
    image.save(output_path)

    print(f"Saved: {output_path}")
    display(image)

    return output_path


# -----------------------------
# CLI entry point
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stable Diffusion generator with memory")
    parser.add_argument("prompt", type=str, help="Text prompt for generation")
    parser.add_argument("--init", type=str, nargs="?", default=None, help="Path to a specific init image (blank = use latest)")
    parser.add_argument("--strength", type=float, default=0.3, help="How much to modify the init image")
    parser.add_argument("--width", type=int, default=512, help="Output image width")
    parser.add_argument("--height", type=int, default=512, help="Output image height")
    parser.add_argument("--scratch", action="store_true", help="Ignore previous image and generate from scratch")
    args = parser.parse_args()

    # If no init image provided, use the most recent one
    if args.scratch:
        init_image = None
    else:
        init_image = args.init or get_latest_image()

    generate_image(
        prompt=args.prompt,
        init_image_path=init_image,
        strength=args.strength,
        width=args.width,
        height=args.height
    )
# %%
