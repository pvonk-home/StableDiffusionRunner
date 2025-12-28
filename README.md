# StableDiffusionRunner
Script to generate images with Stable Diffusion images. Launch targets are available for quick use with prompting in VS Code.
It is designed to run on an PC with an NVIDIA graphics card for optimal Stable Diffusion performance.
After running the script the output image will be in the "outputs" sub-directory.

# Setting up in VS Code
```
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
