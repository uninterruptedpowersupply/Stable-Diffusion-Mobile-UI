#Stable-Diffusion-Mobile-UI
-------------------------------------------------------------------------------------------------------------------

Generate images using Stable Diffusion 1.5 with as little as 1.5 GB of VRAM.
-------------------------------------------------------------------------------------------------------------------
Warning: This application can generate NSFW, gore, and other disturbing material. Please use with caution.
-------------------------------------------------------------------------------------------------------------------
To-Do List
-------------------------------------------------------------------------------------------------------------------
~~Fix crash after saving an image of 512x512 or higher resolution.~~,
~~fix requirements file,~~
Fix SDXL,
create an exe,
Create an APK.

Lower Priority / To Be Addressed Later (or Never):
-------------------------------------------------------------------------------------------------------------------
Add support for Stable Diffusion 3 and Flux when consumer hardware can support them (aware of SD 3.5 variants),

Allow users to paste numbers with commas in the custom resolution menu,

Add vae and lora support,

fix the hight resource requiremnts of SDXL ( 7-6.7gb vram at 512x512 )

High memory usage at resolutions greater than 768x768,

Make it ios compatible.

How to Use
-------------------------------------------------------------------------------------------------------------------
(Windows Only; Might Work with Linux)

Run the RUN_FIRST_TIME_ONLY_.bat file.

Then, run run.bat.

The application requires a peak VRAM usage of 1.5 GB depending on the resolution.

The Forge web UI was run with the following arguments:
-------------------------------------------------------------------------------------------------------------------
--api --pin-shared-memory --cuda-stream --cuda-malloc --xformers

Prompt

positive Trees, snow, evening, glow, bloom, sun set, yellow, orange, 4k, text "STABLE", "DIFFUSION", "MOBILE", "UI", clear tect

negative Low-res

seed 1

steps 20

![Screenshot 2025-02-14 153313](https://github.com/user-attachments/assets/57abd399-d495-4d23-95f0-a22514fca544)
![Screenshot 2025-02-14 153330](https://github.com/user-attachments/assets/7637e51d-b779-43cc-9dd3-122a8d13e65c)
![Screenshot 2025-02-14 155637](https://github.com/user-attachments/assets/77ac6d17-40f5-4dc0-88cc-e24b10da8109)

STABLE DIFFUSION MOBLE UI DOES NOT SUPPORT CUSTOM VAE

----------------------------------------------------------------------------------------------------------------
Models used
----------------------------------------------------------------------------------------------------------------
SD 1.5 - https://civitai.com/models/25694/epicrealism
SDXL - https://civitai.com/models/833294?modelVersionId=1116447



----------------------------------------------------------------------------------------------------------------
SPECIAL THANKS TO 
-------------------------------------------------------------------------------------------------------------------

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
