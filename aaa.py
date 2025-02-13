import os
# Set the environment variable early to help mitigate CUDA fragmentation issues.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys, threading, json, time, gc, warnings, torch
from PIL import Image
from kivy.app import App
from kivy.clock import Clock
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.image import Image as KivyImage
from kivy.uix.progressbar import ProgressBar
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.togglebutton import ToggleButton
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline, DDIMScheduler

# Suppress non-critical warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# --- Debug Stream Setup ---
class DebugStream:
    def __init__(self, original):
        self.original = original
        self.buffer = []
    def write(self, message):
        self.buffer.append(message)
        self.original.write(message)
    def flush(self):
        self.original.flush()
    def get_output(self):
        return ''.join(self.buffer)

debug_stream = DebugStream(sys.stdout)
sys.stdout = debug_stream
sys.stderr = debug_stream

# --- Constants & Setup ---
MODEL_CACHE_FILE = "model_cache.json"
OUTPUT_DIR = "outputs"
RESOLUTIONS = [("128x128", 128), ("256x256", 256), ("512x512", 512), ("768x768", 768)]
STEPS_DEFAULT = 25
CFG_SCALE = 7.5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_unique_filename():
    idx = 1
    while os.path.exists(os.path.join(OUTPUT_DIR, f"output{idx}.png")):
        idx += 1
    return os.path.join(OUTPUT_DIR, f"output{idx}.png")

def load_cache():
    try:
        with open(MODEL_CACHE_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def save_cache(model_path, seed, resolution, steps):
    try:
        cache_data = {"model_path": model_path, "seed": seed, "resolution": resolution, "steps": steps}
        with open(MODEL_CACHE_FILE, "w") as f:
            json.dump(cache_data, f)
    except Exception as e:
        print("Cache error:", e)

def try_enable(func, name):
    """Helper to enable offloading/slicing features with error handling."""
    try:
        func()
        print(f"{name} enabled.")
    except Exception as e:
        print(f"{name} not enabled: {e}")

# --- Image Generation Pipeline with Memory Offloading & SDXL Support ---
def generate_image_with_progress(model_path, prompt, negative_prompt, width, height,
                                 steps, cfg_scale, seed,
                                 progress_callback, cancel_event, use_sdxl=False):
    start_time = time.time()
    pipe = None
    try:
        if use_sdxl:
            # Initialize the SDXL pipeline
            pipe = StableDiffusionXLPipeline.from_single_file(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None  # Disabled for offline use.
            )
        else:
            # Initialize the standard pipeline
            pipe = StableDiffusionPipeline.from_single_file(
                model_path,
                low_cpu_mem_usage=True,
                torch_dtype=torch.float16,
                use_safetensors=True,
                safety_checker=None  # Disabled for offline use.
            )
        try:
            pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
            print("DDIMScheduler enabled.")
        except Exception as e:
            print("Failed to set DDIMScheduler, using default scheduler:", e)

        # Enable aggressive memory-saving techniques.
        try_enable(lambda: pipe.enable_xformers_memory_efficient_attention(), "xFormers Memory Efficient Attention")
        try_enable(lambda: pipe.enable_vae_slicing(), "VAE Slicing")
        try_enable(lambda: pipe.enable_attention_slicing(slice_size=1), "Attention Slicing")
        try_enable(lambda: pipe.enable_sequential_cpu_offload(), "Sequential CPU Offload")
        try_enable(lambda: pipe.enable_model_cpu_offload(), "Model CPU Offload")

        pipe.to(DEVICE)
        progress_callback(0, steps)
        generator = torch.Generator(device=DEVICE).manual_seed(int(seed) if seed.strip() else 1)

        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg_scale,
                width=width,
                height=height,
                generator=generator
            )
            image = result.images[0]

        progress_callback(steps, steps)
        output_path = get_unique_filename()
        image.save(output_path)
        return output_path, time.time() - start_time

    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            print("Generation error: CUDA out of memory. Please try lowering the resolution or use a device with more memory.")
            return None, None
        else:
            print("Generation error:", e)
            return None, None
    finally:
        try:
            del pipe
        except Exception:
            pass
        torch.cuda.empty_cache()
        gc.collect()

# --- Kivy UI Application ---
class SDApp(App):
    def build(self):
        self.title = "Stable Diffusion Mobile UI (Offline)"
        self.model_path = ""
        self.selected_resolution = 512  # Default for preset buttons.
        self.custom_width = None  # Custom resolution width.
        self.custom_height = None  # Custom resolution height.
        self.cancel_event = None
        self.use_sdxl = False  # Flag to toggle SDXL support.

        anchor = AnchorLayout()
        scroll = ScrollView(size_hint=(1, 1))
        main = BoxLayout(orientation="vertical", spacing=10, padding=10, size_hint_y=None)
        main.bind(minimum_height=main.setter("height"))
        scroll.add_widget(main)
        anchor.add_widget(scroll)

        main.add_widget(Label(text="⚠️ Generated images can be controversial. You are responsible for your content.",
                                size_hint=(1, None), height=30, color=(1, 0, 0, 1)))
        self.prompt_input = TextInput(hint_text="Enter positive prompt", size_hint=(1, None), height=80)
        self.neg_prompt_input = TextInput(hint_text="Enter negative prompt", size_hint=(1, None), height=80)
        main.add_widget(self.prompt_input)
        main.add_widget(self.neg_prompt_input)

        model_btn = Button(text="Select Merged Model File (.safetensors)", size_hint=(1, None), height=50,
                           background_color=(0.2, 0.6, 1, 1))
        model_btn.bind(on_press=self.select_model_file)
        main.add_widget(model_btn)

        # SDXL Mode toggle button.
        self.sdxl_toggle = ToggleButton(text="SDXL Mode: OFF", size_hint=(1, None), height=50,
                                        background_color=(0.7, 0.7, 1, 1))
        self.sdxl_toggle.bind(on_press=self.toggle_sdxl)
        main.add_widget(self.sdxl_toggle)

        # Resolution preset buttons.
        res_layout = GridLayout(cols=4, size_hint=(1, None), height=50, spacing=10)
        self.res_checkboxes = {}
        for text, res in RESOLUTIONS:
            btn = Button(text=text, size_hint=(1, None), height=40)
            btn.bind(on_press=lambda inst, r=res: self.set_resolution(r))
            self.res_checkboxes[res] = btn
            res_layout.add_widget(btn)
        main.add_widget(res_layout)
        self.set_resolution(self.selected_resolution)

        # Custom resolution button.
        custom_res_btn = Button(text="Custom Resolution", size_hint=(1, None), height=50,
                                background_color=(0.6, 0.6, 1, 1))
        custom_res_btn.bind(on_press=self.show_custom_resolution_popup)
        main.add_widget(custom_res_btn)

        # Steps slider (10 to 50 steps, default 25).
        steps_layout = BoxLayout(orientation="vertical", size_hint=(1, None), height=60)
        steps_layout.add_widget(Label(text="Steps (10 to 50):", size_hint=(1, None), height=20))
        self.steps_slider = Slider(min=10, max=50, value=STEPS_DEFAULT, size_hint=(1, None), height=30)
        self.steps_value_label = Label(text=str(STEPS_DEFAULT), size_hint=(1, None), height=20)
        self.steps_slider.bind(value=self.update_steps_label)
        steps_layout.add_widget(self.steps_slider)
        steps_layout.add_widget(self.steps_value_label)
        main.add_widget(steps_layout)

        seed_layout = BoxLayout(orientation="vertical", size_hint=(1, None), height=60)
        seed_layout.add_widget(Label(text="Seed:", size_hint=(1, None), height=20))
        self.seed_input = TextInput(hint_text="1", multiline=False, size_hint=(1, None), height=40)
        seed_layout.add_widget(self.seed_input)
        main.add_widget(seed_layout)

        self.progress_bar = ProgressBar(max=STEPS_DEFAULT, value=0, size_hint=(1, None), height=30)
        self.progress_label = Label(text="Progress: 0/" + str(STEPS_DEFAULT), size_hint=(1, None), height=30)
        main.add_widget(self.progress_label)
        main.add_widget(self.progress_bar)

        # Button layout for generate, cancel, and debug output.
        btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=60, spacing=10)
        self.generate_btn = Button(text="Generate Image", background_color=(0.3, 0.8, 0.3, 1))
        self.cancel_btn = Button(text="Cancel", background_color=(0.8, 0.3, 0.3, 1), disabled=True)
        self.generate_btn.bind(on_press=self.start_generation)
        self.cancel_btn.bind(on_press=self.cancel_generation)
        btn_layout.add_widget(self.generate_btn)
        btn_layout.add_widget(self.cancel_btn)
        main.add_widget(btn_layout)

        # Button to show debug output.
        debug_btn = Button(text="Show Debug Output", size_hint=(1, None), height=50, background_color=(0.8, 0.8, 0.3, 1))
        debug_btn.bind(on_press=self.show_debug_output)
        main.add_widget(debug_btn)

        # Display area for the generated image.
        self.image_display = KivyImage(size_hint=(1, 0.5))
        main.add_widget(self.image_display)
        # Label to display generation time.
        self.gen_time_label = Label(text="Generation time: N/A", size_hint=(1, None), height=30)
        main.add_widget(self.gen_time_label)

        cached = load_cache()
        if cached:
            self.model_path = cached.get("model_path", "")
            self.selected_resolution = cached.get("resolution", 512)
            self.steps_slider.value = cached.get("steps", STEPS_DEFAULT)
            self.seed_input.text = str(cached.get("seed", ""))
            self.progress_bar.max = cached.get("steps", STEPS_DEFAULT)
            self.progress_label.text = "Progress: 0/" + str(cached.get("steps", STEPS_DEFAULT))
        else:
            self.progress_bar.max = STEPS_DEFAULT

        Clock.schedule_interval(lambda dt: gc.collect(), 15)
        return anchor

    def update_steps_label(self, instance, value):
        self.steps_value_label.text = str(int(value))

    def toggle_sdxl(self, instance):
        # Toggle the SDXL mode flag.
        self.use_sdxl = not self.use_sdxl
        instance.text = "SDXL Mode: ON" if self.use_sdxl else "SDXL Mode: OFF"
        print("SDXL mode set to:", self.use_sdxl)

    def set_resolution(self, res):
        self.selected_resolution = res
        self.custom_width = None
        self.custom_height = None
        for r, btn in self.res_checkboxes.items():
            btn.background_color = (0.2, 0.8, 0.2, 1) if r == res else (1, 1, 1, 1)
        print("Selected preset resolution:", res)

    def show_custom_resolution_popup(self, instance):
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        width_input = TextInput(hint_text="Enter width (e.g., 512)", multiline=False, input_filter='int')
        height_input = TextInput(hint_text="Enter height (e.g., 512)", multiline=False, input_filter='int')
        submit_btn = Button(text="Set Custom Resolution", size_hint=(1, None), height=40)
        content.add_widget(Label(text="Custom Resolution"))
        content.add_widget(width_input)
        content.add_widget(height_input)
        content.add_widget(submit_btn)
        popup = Popup(title="Custom Resolution", content=content, size_hint=(0.8, 0.5))
        
        def set_custom_res(instance):
            try:
                self.custom_width = int(width_input.text)
                self.custom_height = int(height_input.text)
                print(f"Custom resolution set to: {self.custom_width}x{self.custom_height}")
                popup.dismiss()
            except ValueError:
                self.show_error("Please enter valid integer values for width and height.")
        
        submit_btn.bind(on_press=set_custom_res)
        popup.open()

    def select_model_file(self, instance):
        filechooser = FileChooserListView(filters=["*.safetensors"], path=os.getcwd())
        self.popup = Popup(title="Select Merged Model File", content=filechooser, size_hint=(0.9, 0.8))
        filechooser.bind(on_submit=self.on_model_selected)
        self.popup.open()

    def on_model_selected(self, instance, selection, touch):
        if selection:
            self.model_path = selection[0]
            print("Selected model:", self.model_path)
        self.popup.dismiss()
        self.popup = None

    def start_generation(self, instance):
        if not self.model_path:
            self.show_error("Please select a model file first!")
            return
        if not self.prompt_input.text.strip():
            self.show_error("Please enter a prompt!")
            return
        self.generate_btn.disabled = True
        self.cancel_btn.disabled = False
        steps = int(self.steps_slider.value)
        self.progress_bar.value = 0
        self.progress_bar.max = steps
        self.progress_label.text = "Progress: 0/" + str(steps)
        self.cancel_event = threading.Event()
        threading.Thread(target=self.run_generation, daemon=True).start()

    def run_generation(self):
        try:
            steps = int(self.steps_slider.value)
            seed = self.seed_input.text.strip() or "1"
            if self.custom_width and self.custom_height:
                width, height = self.custom_width, self.custom_height
            else:
                width = height = self.selected_resolution
            output_path, gen_time = generate_image_with_progress(
                self.model_path,
                self.prompt_input.text,
                self.neg_prompt_input.text,
                width,
                height,
                steps,
                CFG_SCALE,
                seed,
                self.update_progress,
                self.cancel_event,
                use_sdxl=self.use_sdxl
            )
            print(f"run_generation: output_path = {output_path}")
            if output_path:
                Clock.schedule_once(lambda dt: self.update_image(output_path, gen_time))
                save_cache(self.model_path, seed, width, steps)
            else:
                print("Output path is None, not updating image.")
        except Exception as e:
            Clock.schedule_once(lambda dt: self.show_error(str(e)))
        finally:
            Clock.schedule_once(lambda dt: self.reset_buttons())

    def update_progress(self, step, total_steps):
        def update_ui(dt):
            self.progress_bar.value = step
            self.progress_label.text = f"Progress: {step}/{total_steps}"
        Clock.schedule_once(update_ui)

    def update_image(self, img_path, gen_time):
        self.image_display.source = img_path
        self.image_display.reload()
        self.gen_time_label.text = f"Generation time: {gen_time:.2f} sec"

    def reset_buttons(self, dt=None):
        self.generate_btn.disabled = False
        self.cancel_btn.disabled = True

    def cancel_generation(self, instance):
        if self.cancel_event:
            self.cancel_event.set()
        self.reset_buttons()
        print("Generation cancelled.")

    def show_error(self, message):
        popup = Popup(title="Error", content=Label(text=message), size_hint=(0.7, 0.4))
        popup.open()

    def show_info(self, message):
        popup = Popup(title="Info", content=Label(text=message), size_hint=(0.7, 0.4))
        popup.open()

    def show_debug_output(self, instance):
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        debug_text = TextInput(text=debug_stream.get_output(), readonly=True, multiline=True)
        close_btn = Button(text="Close", size_hint=(1, None), height=40)
        content.add_widget(debug_text)
        content.add_widget(close_btn)
        popup = Popup(title="Debug Output", content=content, size_hint=(0.9, 0.9))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()

if __name__ == "__main__":
    SDApp().run()
