#Stable-Diffusion-Mobile-UI
-------------------------------------------------------------------------------------------------------------------

Generate images using Stable Diffusion 1.5 with as little as 2.1 GB of VRAM.
-------------------------------------------------------------------------------------------------------------------
Warning: This application can generate NSFW, gore, and other disturbing material. Please use with caution.
-------------------------------------------------------------------------------------------------------------------
To-Do List:
-------------------------------------------------------------------------------------------------------------------
~~Fix crash after saving an image of 512x512 or higher resolution.~~,
~~fix requirements file,~~
Fix SDXL,
create an exe,
Create an APK.

Lower Priority / To Be Addressed Later (or Never):
-------------------------------------------------------------------------------------------------------------------
~~High memory usage at resolutions greater than 768x768~~,(2.5gb peek usage at 1024x1024)

Add support for Stable Diffusion 3 and Flux when consumer hardware can support them (aware of SD 3.5 variants),

Allow users to paste numbers with commas in the custom resolution menu,

Add vae and lora support,

fix the hight resource requiremnts of SDXL ( 5.5gb vram at 512x512 )

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

![Screenshot 2025-04-08 111732](https://github.com/user-attachments/assets/1dda2294-1699-42d8-83d2-bb9425c41e61)
![image](https://github.com/user-attachments/assets/0433644a-8112-442d-9736-131fb005abdc)
![Screenshot 2025-02-14 155632](https://github.com/user-attachments/assets/a98a1406-9458-4739-bf77-eb22b5440abb)
![Screenshot 2025-02-14 155637](https://github.com/user-attachments/assets/77ac6d17-40f5-4dc0-88cc-e24b10da8109)
New collage mode

AT 1024X1024
![Screenshot 2025-04-08 112036](https://github.com/user-attachments/assets/2e64bde2-d108-4087-99ba-d8a472be8d45)
![image](https://github.com/user-attachments/assets/4c2cd82a-0f2a-4411-95b3-79ffa6b309d1)

AT 128X128
![Screenshot 2025-04-08 112257](https://github.com/user-attachments/assets/2aaa75d0-d7b4-46ff-8658-fe170c815853)
![image](https://github.com/user-attachments/assets/5a531044-7d40-42c1-9c4e-f2bcb2df364c)

STABLE DIFFUSION MOBLE UI DOES NOT SUPPORT CUSTOM VAE, TEXT ENCODERS OR LORA YET

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
