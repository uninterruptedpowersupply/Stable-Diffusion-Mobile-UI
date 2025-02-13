Stable-Diffusion-Mobile-UI
Generate images using Stable Diffusion 1.5 with as little as 2.5 GB of VRAM.

Warning: This application can generate NSFW, gore, and other disturbing material. Please use with caution.

To-Do List
Fix crash after saving an image of 512x512 or higher resolution.
Create an APK.
Lower Priority / To Be Addressed Later (or Never):

Add support for Stable Diffusion 3 and Flux when consumer hardware can support them (aware of SD 3.5 variants).
Allow users to paste numbers with commas in the custom resolution menu.
How to Use (Windows Only; Might Work with Linux)
Run the RUN_FIRST_TIME_ONLY_.bat file.
Then, run run.bat.
The application requires a peak VRAM usage of 2.5 GB.

The Forge web UI was run with the following arguments:

css
Copy
Edit
--api --pin-shared-memory --cuda-stream --cuda-malloc --xformers

![image](https://github.com/user-attachments/assets/c632e0d8-613d-41c8-bdb7-385208eb49f4)
![Screenshot 2025-02-12 194938](https://github.com/user-attachments/assets/4bd0779c-f8d4-49a0-b0b6-d99981c41f1e)
![image](https://github.com/user-attachments/assets/73180a6e-aa52-47f7-97d0-54a822a9de65)

----------------------------------------------------------------------------------------------------------------
SPECIAL THANKS TO 

https://github.com/AUTOMATIC1111/stable-diffusion-webui
https://github.com/Panchovix/stable-diffusion-webui-reForge
https://github.com/lllyasviel/stable-diffusion-webui-forge
https://github.com/comfyanonymous/ComfyUI
https://github.com/cumulo-autumn/StreamDiffusion
https://github.com/huggingface/diffusers
https://pytorch.org/
https://kivy.org/
https://python-pillow.org/
https://numpy.org/
https://github.com/facebookresearch/xformers
https://arxiv.org/abs/2112.10752
https://arxiv.org/abs/2010.02502
