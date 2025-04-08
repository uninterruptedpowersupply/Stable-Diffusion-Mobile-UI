# pipeline_manager.py
import time, gc, torch, os
import math
from PIL import Image
from tqdm import tqdm # Progress bars
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler
from diffusers.utils import logging, deprecate

# Local imports
from memory_utils import memory_cleanup, get_memory_usage
from deepseek_optimizations import apply_deepseek_optimizations

# --- Setup Flash Attention / xFormers ---
# (Keep the previous logic)
HAS_FLASH_ATTENTION_V2 = False
attn_processor_to_use = None
try:
    import flash_attn
    if hasattr(flash_attn, '__version__') and int(flash_attn.__version__.split('.')[0]) >= 2:
        from diffusers.models.attention_processor import AttnProcessor2_0 as AttnProcessor
        try:
             from diffusers.models.attention_processor import FlashAttnProcessor2_0
             attn_processor_to_use = FlashAttnProcessor2_0()
             HAS_FLASH_ATTENTION_V2 = True
             print("Flash Attention 2 processor found.")
        except ImportError:
            print("FlashAttnProcessor2_0 not found, using default AttnProcessor2_0.")
            attn_processor_to_use = AttnProcessor()
            HAS_FLASH_ATTENTION_V2 = False
    else:
        print("Flash Attention V1 found or version < 2, attempting xFormers.")
        HAS_FLASH_ATTENTION_V2 = False
except ImportError as e:
    print(f"Flash Attention not found ({e}). Will attempt xFormers.")
    HAS_FLASH_ATTENTION_V2 = False

# --- Constants ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
logging.set_verbosity_error()


# --- Pipeline Manager Class ---
class PipelineManager:
    # --- __init__, unload_model, load_model ---
    # --- _get_text_embeddings, _prepare_latents ---
    # (Keep these methods exactly as in the previous version)
    def __init__(self):
        self.pipe = None
        self.model_path = None
        self.use_sdxl = False
        self.current_mode = "Normal"
        self.tile_size = 96
        self.tile_overlap = 32

    def unload_model(self):
        print("Unloading model...")
        if self.pipe is not None:
            try:
                if not hasattr(self.pipe, "_hf_hook"):
                   print("Manually moving pipe to CPU...")
                   self.pipe.to('cpu')
                else:
                   print("Accelerate should handle CPU offload cleanup.")
            except Exception as e:
                print(f"Warning: Error moving pipe to CPU before deletion: {e}")
            del self.pipe
            self.pipe = None
        self.model_path = None
        memory_cleanup()
        print("Model unloaded.")
        print(get_memory_usage("Current"))

    def load_model(self, model_path, use_sdxl):
        start_time = time.time()
        if self.pipe is not None and (self.model_path != model_path or self.use_sdxl != use_sdxl):
            self.unload_model()

        if self.pipe is None:
            print(f"Loading model: {os.path.basename(model_path)} (SDXL: {use_sdxl})")
            self.model_path = model_path
            self.use_sdxl = use_sdxl
            pipeline_class = StableDiffusionXLPipeline if use_sdxl else StableDiffusionPipeline

            try:
                self.pipe = pipeline_class.from_single_file(
                    model_path, torch_dtype=DTYPE, use_safetensors=True,
                    safety_checker=None, low_cpu_mem_usage=True,
                    variant="fp16" if DTYPE == torch.float16 else None
                )
                print("Pipeline base loaded.")
                print(get_memory_usage("Current"))

                try: # Scheduler
                    self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)
                    print("DDIMScheduler enabled.")
                except Exception as e: print(f"Could not set DDIMScheduler, using default: {e}")

                optim_applied = False # Attention
                if HAS_FLASH_ATTENTION_V2 and attn_processor_to_use:
                    try:
                        self.pipe.unet.set_attn_processor(attn_processor_to_use)
                        print("Flash Attention 2 enabled."); optim_applied = True
                    except Exception as e: print(f"Flash Attention 2 failed: {e}. Falling back...")
                if not optim_applied:
                    try:
                        self.pipe.enable_xformers_memory_efficient_attention()
                        print("xFormers enabled."); optim_applied = True
                    except Exception as e:
                        print(f"xFormers failed: {e}. Falling back to Attention Slicing.")
                        try:
                            self.pipe.enable_attention_slicing(slice_size=1)
                            print("Attention Slicing enabled (Fallback)."); optim_applied = True
                        except Exception as ex: print(f"Attention slicing failed: {ex}.")

                try: # VAE Opts
                    self.pipe.enable_vae_tiling(); print("VAE Tiling enabled.")
                except AttributeError:
                     try: self.pipe.enable_vae_slicing(); print("VAE Slicing enabled.")
                     except Exception as e: print(f"VAE Slicing not available: {e}")
                except Exception as e: print(f"VAE Tiling failed: {e}")

                offload_applied = False # CPU Offload
                try:
                    self.pipe.enable_model_cpu_offload()
                    print("Model CPU Offload enabled."); offload_applied = True
                except Exception as e: print(f"Model CPU Offload failed: {e}.")

                apply_deepseek_optimizations(self.pipe) # DeepSeek

                if not offload_applied: # Device Placement
                     print(f"Moving pipeline to {DEVICE}...")
                     self.pipe.to(DEVICE)
                     print(f"Pipeline on {DEVICE}.")
                else: print("CPU Offload handles device placement.")

                print(f"Pipeline configured in {time.time() - start_time:.2f} seconds.")
                print(get_memory_usage("Peak"))
                return True

            except Exception as e:
                import traceback
                print(f"FATAL: Pipeline initialization failed: {e}\n{traceback.format_exc()}")
                self.pipe = None; self.model_path = None
                memory_cleanup()
                return False
        else:
            print("Model already loaded."); return True

    def _get_text_embeddings(self, prompt, negative_prompt, device):
        num_images_per_prompt = 1
        do_classifier_free_guidance = True
        (prompt_embeds, negative_prompt_embeds,
         pooled_prompt_embeds, negative_pooled_prompt_embeds) = self.pipe.encode_prompt(
             prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt,
             do_classifier_free_guidance=do_classifier_free_guidance, negative_prompt=negative_prompt,
         )
        return prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds

    def _prepare_latents(self, width, height, generator, device):
        vae_scale_factor = self.pipe.vae_scale_factor
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor
        num_channels_latents = self.pipe.unet.config.in_channels
        return self.pipe.prepare_latents(
                1, num_channels_latents, latent_height, latent_width,
                DTYPE, device, generator,
            )

    # --- Tiling Generation Method (Keep as is) ---
    def _generate_tiled_coherent(self, prompt, negative_prompt, width, height, steps, cfg_scale, generator, progress_callback=None):
        vae_scale_factor = self.pipe.vae_scale_factor
        latent_height = height // vae_scale_factor
        latent_width = width // vae_scale_factor

        if latent_width < self.tile_size or latent_height < self.tile_size:
            print(f"WARNING: Tiled mode fallback to Normal for {width}x{height} (latent {latent_width}x{latent_height} vs tile {self.tile_size}x{self.tile_size}).")
            normal_params = {
                "prompt": prompt, "negative_prompt": negative_prompt,
                "width": width, "height": height,
                "num_inference_steps": steps, "guidance_scale": cfg_scale,
                "generator": generator,
                 "callback_steps": 1,
                 "callback": lambda step, t, latents: progress_callback(int(step), steps) if progress_callback else None # Pass callback correctly
            }
            # Ensure result access is safe
            result = self.pipe(**normal_params)
            return result.images[0] if result.images else None

        print(f"Starting Tiled Coherent Generation ({width}x{height})")
        print(f"Latent Tile Size: {self.tile_size}x{self.tile_size}, Overlap: {self.tile_overlap}")
        current_device = generator.device

        num_channels_latents = self.pipe.unet.config.in_channels
        prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = self._get_text_embeddings(
            prompt, negative_prompt, current_device
        )

        if cfg_scale > 1.0:
             if negative_prompt_embeds is None: raise ValueError("Negative prompt embeddings required for CFG > 1.0.")
             text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds])
             if self.use_sdxl: add_text_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds])
        else:
             text_embeddings = prompt_embeds
             if self.use_sdxl: add_text_embeds = pooled_prompt_embeds

        latents = self._prepare_latents(width, height, generator, current_device)
        self.pipe.scheduler.set_timesteps(steps, device=current_device)
        timesteps = self.pipe.scheduler.timesteps
        latents = latents * self.pipe.scheduler.init_noise_sigma

        stride = self.tile_size - self.tile_overlap
        num_tiles_h = math.ceil((latent_height - self.tile_overlap) / stride)
        num_tiles_w = math.ceil((latent_width - self.tile_overlap) / stride)
        total_tiles = num_tiles_h * num_tiles_w
        print(f"Processing {num_tiles_h}x{num_tiles_w} = {total_tiles} tiles per step.")

        num_warmup_steps = len(timesteps) - steps * self.pipe.scheduler.order
        with tqdm(total=steps, desc="Tiled Diffusion Steps") as pbar_steps:
            for i, t in enumerate(timesteps):
                next_latents_processed = torch.zeros_like(latents, device=current_device)
                weights_sum = torch.zeros_like(latents, device=current_device)
                latents_input = latents

                tile_idx = 0
                for h_idx in range(num_tiles_h):
                    for w_idx in range(num_tiles_w):
                        tile_idx += 1
                        y_start = h_idx * stride
                        x_start = w_idx * stride
                        y_end = min(y_start + self.tile_size, latent_height)
                        x_end = min(x_start + self.tile_size, latent_width)
                        y_start = max(0, y_end - self.tile_size)
                        x_start = max(0, x_end - self.tile_size)

                        tile_latent = latents_input[:, :, y_start:y_end, x_start:x_end]

                        latent_model_input = torch.cat([tile_latent] * 2) if cfg_scale > 1.0 else tile_latent
                        latent_model_input = self.pipe.scheduler.scale_model_input(latent_model_input, t)

                        unet_payload = {"sample": latent_model_input, "timestep": t, "encoder_hidden_states": text_embeddings}
                        if self.use_sdxl:
                             add_time_ids = self.pipe._get_add_time_ids((height, width), (0,0), (height, width), dtype=prompt_embeds.dtype, device=current_device)
                             add_time_ids = torch.cat([add_time_ids] * 2) if cfg_scale > 1.0 else add_time_ids
                             unet_payload["added_cond_kwargs"] = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}

                        noise_pred = self.pipe.unet(**unet_payload).sample

                        if cfg_scale > 1.0:
                            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_text - noise_pred_uncond)

                        tile_latents_next = self.pipe.scheduler.step(noise_pred, t, tile_latent, return_dict=False)[0]
                        next_latents_processed[:, :, y_start:y_end, x_start:x_end] += tile_latents_next
                        weights_sum[:, :, y_start:y_end, x_start:x_end] += 1.0

                        if progress_callback:
                             progress = (i + tile_idx / total_tiles) / len(timesteps)
                             progress_callback(int(progress * steps), steps)

                weights_sum = torch.where(weights_sum == 0, torch.ones_like(weights_sum), weights_sum)
                latents = next_latents_processed / weights_sum
                pbar_steps.set_postfix({"VRAM": f"{get_memory_usage('Current')}"}) # Renamed postfix key
                pbar_steps.update()

        print(f"Decoding final latents... {get_memory_usage('Current')}")
        if not hasattr(self.pipe, "_hf_hook"): self.pipe.vae.to(current_device)

        latents = 1 / self.pipe.vae.config.scaling_factor * latents
        image = self.pipe.vae.decode(latents.to(self.pipe.vae.device, dtype=self.pipe.vae.dtype), return_dict=False)[0]
        image = self.pipe.image_processor.postprocess(image, output_type="pil", do_denormalize=[True] * image.shape[0])

        if not hasattr(self.pipe, "_hf_hook"): self.pipe.vae.to('cpu')
        print("Tiled Coherent Generation Finished.")
        return image[0]


    # --- Main Generation Function (Revised Call to Tiled) ---
    def generate_image(self, prompt, negative_prompt, width, height, steps, cfg_scale, seed, mode="Normal", progress_callback=None):
        if self.pipe is None: raise RuntimeError("Pipeline not loaded.")

        self.current_mode = mode
        print(f"\n--- Starting Generation ---")
        print(f"Mode: {mode}, Model: {os.path.basename(self.model_path)}, SDXL: {self.use_sdxl}")
        print(f"Res: {width}x{height}, Steps: {steps}, CFG: {cfg_scale}, Seed: {seed}")
        print(f"Initial state: {get_memory_usage('Peak')}")
        start_time = time.time()

        try:
            seed_str = str(seed).strip(); seed_int = int(seed_str) if seed_str else int(time.time())
        except ValueError:
            print(f"Invalid seed '{seed}', using current time."); seed_int = int(time.time())

        generator_device = "cpu" if hasattr(self.pipe, "_hf_hook") and self.pipe._hf_hook is not None else DEVICE
        generator = torch.Generator(device=generator_device).manual_seed(seed_int)
        print(f"Using seed: {seed_int} on device: {generator_device}")

        final_image = None
        effective_mode = mode

        try:
            with torch.no_grad():
                if progress_callback: progress_callback(0, steps)

                if mode == "Tiled":
                     effective_mode = "Tiled (Coherent)"
                     # --- CORRECTED CALL ---
                     # Pass only the expected arguments explicitly
                     final_image = self._generate_tiled_coherent(
                         prompt=prompt,
                         negative_prompt=negative_prompt,
                         width=width,
                         height=height,
                         steps=steps,              # Pass 'steps' not 'num_inference_steps'
                         cfg_scale=cfg_scale,
                         generator=generator,
                         progress_callback=progress_callback
                     )
                     # --- END CORRECTION ---

                elif mode == "Normal":
                    print("Generating in Normal mode...")
                    effective_mode = "Normal"
                    result = self.pipe(
                        prompt=prompt, negative_prompt=negative_prompt,
                        width=width, height=height,
                        num_inference_steps=steps, guidance_scale=cfg_scale,
                        generator=generator,
                        callback_steps=1,
                        callback=lambda step, t, latents: progress_callback(int(step), steps) if progress_callback else None
                    )
                    final_image = result.images[0] if result.images else None

                elif mode == "Collage":
                    # (Collage logic remains the same)
                    print("Generating in Collage mode (2x2 grid)...")
                    effective_mode = "Collage"
                    grid_size = 2
                    tile_w, tile_h = width // grid_size, height // grid_size
                    if tile_w < 64 or tile_h < 64: raise ValueError("Collage requires base resolution >= 128x128.")
                    images = []
                    total_tiles = grid_size * grid_size
                    for i in range(total_tiles):
                        tile_seed = seed_int + i
                        tile_generator = torch.Generator(device=generator.device).manual_seed(tile_seed)
                        print(f"  Generating collage tile {i+1}/{total_tiles} (Seed: {tile_seed})...")
                        tile_image = self.pipe(
                            prompt=prompt, negative_prompt=negative_prompt,
                            width=tile_w, height=tile_h,
                            num_inference_steps=steps, guidance_scale=cfg_scale,
                            generator=tile_generator
                        ).images[0]
                        images.append(tile_image)
                        if progress_callback: progress_callback(int(steps * (i + 1) / total_tiles), steps)
                        memory_cleanup()
                    print("Combining collage tiles...")
                    final_image = Image.new('RGB', (width, height))
                    for idx, img in enumerate(images):
                        row = idx // grid_size; col = idx % grid_size
                        final_image.paste(img, (col * tile_w, row * tile_h))
                else:
                    raise ValueError(f"Unknown generation mode: {mode}")

                if progress_callback: progress_callback(steps, steps)

        except Exception as e:
            import traceback
            print(f"!!! Generation failed during '{effective_mode}' mode: {e}\n{traceback.format_exc()}")
            memory_cleanup(); raise

        finally:
            elapsed_time = time.time() - start_time
            print(f"Generation finished in {elapsed_time:.2f} seconds.")
            print(f"Final state: {get_memory_usage('Peak')}")
            memory_cleanup()

        return final_image, elapsed_time