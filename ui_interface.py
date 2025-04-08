# ui_interface.py
import os
import json
import threading
import time
import gc
import traceback # Import traceback

from kivy.app import App
from kivy.clock import Clock
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.popup import Popup
from kivy.uix.slider import Slider
from kivy.uix.filechooser import FileChooserListView
from kivy.uix.togglebutton import ToggleButton
from kivy.uix.progressbar import ProgressBar
from kivy.uix.image import Image as KivyImage
from kivy.graphics.texture import Texture
from kivy.uix.dropdown import DropDown
from PIL import Image

# Local imports
try:
    from pipeline_manager import PipelineManager
    from memory_utils import memory_cleanup, get_memory_usage
    from debug_stream import get_debug_output
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Ensure pipeline_manager.py, memory_utils.py, and debug_stream.py are present.")
    exit()


# --- Constants ---
OUTPUT_DIR = "outputs"
MODEL_CACHE_FILE = "model_cache.json"
STEPS_DEFAULT = 20
CFG_SCALE = 7.0
RESOLUTIONS = [("128x128", 128), ("256x256", 256), ("512x512", 512), ("768x768", 768)]
DEFAULT_RESOLUTION_VALUE = 512 if 512 in [r[1] for r in RESOLUTIONS] else RESOLUTIONS[0][1]


# --- Helper Functions (Defined BEFORE the App class) ---
def load_cache():
    """Loads the model cache data from a JSON file."""
    if not os.path.exists(MODEL_CACHE_FILE):
        return {}
    try:
        with open(MODEL_CACHE_FILE, "r") as f:
            content = f.read()
            if not content: return {}
            return json.loads(content)
    except json.JSONDecodeError:
        print(f"Warning: Cache file '{MODEL_CACHE_FILE}' contains invalid JSON.")
        return {}
    except Exception as e:
        print(f"Warning: Could not load cache file '{MODEL_CACHE_FILE}': {e}")
        return {}

def save_cache(data):
    """Saves data to the model cache JSON file."""
    try:
        with open(MODEL_CACHE_FILE, "w") as f:
            json.dump(data, f, indent=4)
    except Exception as e:
        print(f"Cache save error: {e}")


# --- Main Kivy Application Class ---
class SDAppInterface(App):

    def build(self):
        self.title = "Optimized Stable Diffusion UI"
        self.pipeline_manager = PipelineManager()
        # Load cache data immediately, but widget updates happen later
        self.current_cache = load_cache()
        # Initialize state attributes with defaults BEFORE loading from cache
        # This ensures they always exist.
        self.model_path = ""
        self.use_sdxl = False
        self.selected_resolution = DEFAULT_RESOLUTION_VALUE
        self.custom_width = None
        self.custom_height = None
        self.selected_mode = "Normal"
        self.prompt_input_text = "" # Use temporary vars for text inputs if needed
        self.neg_prompt_input_text = ""
        self.seed_text = "1"
        self.steps_value = STEPS_DEFAULT
        # Initialize thread/event variables
        self.generation_thread = None
        self.cancel_requested = threading.Event()

        # --- UI Setup ---
        anchor = AnchorLayout()
        scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        main = BoxLayout(orientation="vertical", spacing=8, padding=10, size_hint_y=None)
        main.bind(minimum_height=main.setter("height"))
        scroll.add_widget(main)
        anchor.add_widget(scroll)

        # Model & Mode Selection
        model_box = BoxLayout(size_hint=(1, None), height=40, spacing=5)
        model_btn = Button(text="Select Model (.safetensors)", size_hint=(0.7, 1))
        model_btn.bind(on_press=self.select_model_file)
        self.sdxl_toggle = ToggleButton(text="SDXL: OFF", size_hint=(0.3, 1), group='sd_type', state='normal')
        self.sdxl_toggle.bind(state=self.on_sdxl_toggle)
        model_box.add_widget(model_btn)
        model_box.add_widget(self.sdxl_toggle)
        main.add_widget(model_box)
        self.model_label = Label(text="No model loaded", size_hint=(1, None), height=20, font_size='12sp')
        main.add_widget(self.model_label)

        # Prompt Inputs
        main.add_widget(Label(text="Positive Prompt:", size_hint=(1, None), height=20, halign='left'))
        self.prompt_input = TextInput(hint_text="Enter positive prompt", size_hint=(1, None), height=70, multiline=True)
        main.add_widget(self.prompt_input)
        main.add_widget(Label(text="Negative Prompt:", size_hint=(1, None), height=20, halign='left'))
        self.neg_prompt_input = TextInput(hint_text="Enter negative prompt", size_hint=(1, None), height=50, multiline=True)
        main.add_widget(self.neg_prompt_input)

        # Generation Settings
        settings_grid = BoxLayout(orientation='horizontal', size_hint=(1, None), height=50, spacing=10)
        # Resolution Dropdown
        self.resolution_dropdown = DropDown()
        for text, res_val in RESOLUTIONS:
             btn = Button(text=text, size_hint_y=None, height=40)
             btn.bind(on_release=lambda btn_instance, r=res_val, t=text: self.set_resolution(r, t))
             self.resolution_dropdown.add_widget(btn)
        self.resolution_button = Button(text='Resolution')
        self.resolution_button.bind(on_release=self.resolution_dropdown.open)
        settings_grid.add_widget(self.resolution_button)

        # Generation Mode Dropdown
        self.mode_dropdown = DropDown()
        self.available_modes = ["Normal", "Collage", "Tiled"]
        for mode_name in self.available_modes:
            btn = Button(text=mode_name, size_hint_y=None, height=40)
            btn.bind(on_release=lambda btn_instance, m=mode_name: self.set_generation_mode(m))
            self.mode_dropdown.add_widget(btn)
        self.mode_button = Button(text='Mode: Normal')
        self.mode_button.bind(on_release=self.mode_dropdown.open)
        settings_grid.add_widget(self.mode_button)
        main.add_widget(settings_grid)

        # Custom Resolution Button
        custom_btn = Button(text="Custom Res", size_hint=(1, None), height=40)
        custom_btn.bind(on_press=self.show_custom_resolution_popup)
        main.add_widget(custom_btn)
        self.custom_res_label = Label(text="", size_hint=(1, None), height=20, font_size='12sp')
        main.add_widget(self.custom_res_label)

        # Steps Slider
        steps_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=30)
        steps_layout.add_widget(Label(text="Steps:", size_hint=(0.2, 1)))
        self.steps_slider = Slider(min=10, max=50, value=STEPS_DEFAULT, step=1, size_hint=(0.7, 1))
        self.steps_value_label = Label(text=str(STEPS_DEFAULT), size_hint=(0.1, 1))
        self.steps_slider.bind(value=self.update_steps_label)
        steps_layout.add_widget(self.steps_slider)
        steps_layout.add_widget(self.steps_value_label)
        main.add_widget(steps_layout)

        # Seed Input
        seed_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=40)
        seed_layout.add_widget(Label(text="Seed:", size_hint=(0.2, 1)))
        self.seed_input = TextInput(text="1", multiline=False, size_hint=(0.8, 1), input_filter='int')
        seed_layout.add_widget(self.seed_input)
        main.add_widget(seed_layout)

        # Progress & Control Buttons
        self.progress_bar = ProgressBar(max=STEPS_DEFAULT, value=0, size_hint=(1, None), height=15)
        self.progress_label = Label(text=f"Progress: 0/{STEPS_DEFAULT}", size_hint=(1, None), height=20, font_size='12sp')
        main.add_widget(self.progress_label)
        main.add_widget(self.progress_bar)

        btn_layout = BoxLayout(orientation="horizontal", size_hint=(1, None), height=50, spacing=10)
        self.generate_btn = Button(text="Generate", background_color=(0.2, 0.6, 0.2, 1))
        self.cancel_btn = Button(text="Cancel", background_color=(0.8, 0.2, 0.2, 1), disabled=True)
        self.generate_btn.bind(on_press=self.start_generation)
        self.cancel_btn.bind(on_press=self.cancel_generation)
        btn_layout.add_widget(self.generate_btn)
        btn_layout.add_widget(self.cancel_btn)
        main.add_widget(btn_layout)

        # Image Display & Info
        self.image_display = KivyImage(texture=None, allow_stretch=True, keep_ratio=True, size_hint=(1, None), height=300)
        main.add_widget(self.image_display)
        self.info_label = Label(text="Load model and enter prompt.", size_hint=(1, None), height=30, font_size='12sp')
        main.add_widget(self.info_label)

        # Debug Button
        debug_btn = Button(text="Show Debug Log", size_hint=(1, None), height=40)
        debug_btn.bind(on_press=self.show_debug_output)
        main.add_widget(debug_btn)

        # Schedule post-build initialization
        Clock.schedule_once(self.post_build_init, 0.1)

        Clock.schedule_interval(lambda dt: gc.collect(), 60)
        return anchor

    def post_build_init(self, dt):
        """Initialization steps after the UI is built and widgets exist."""
        print("Post-build initialization...")
        # Load settings from cache into instance variables
        self.load_settings_from_cache()

        # Now update the UI widgets based on the loaded instance variables
        self.prompt_input.text = self.prompt_input_text
        self.neg_prompt_input.text = self.neg_prompt_input_text
        self.seed_input.text = self.seed_text
        self.steps_slider.value = self.steps_value
        self.model_label.text = os.path.basename(self.model_path) if self.model_path else "No model loaded"

        # Update resolution button and custom label
        if self.custom_width and self.custom_height:
             res_text = "Custom Res"
             self.custom_res_label.text = f"Custom: {self.custom_width}x{self.custom_height}"
        else:
             # selected_resolution should now be guaranteed to have a valid value
             res_text = next((text for text, val in RESOLUTIONS if val == self.selected_resolution), RESOLUTIONS[0][0])
             self.custom_res_label.text = ""
        self.resolution_button.text = res_text

        # Update mode button
        self.mode_button.text = f"Mode: {self.selected_mode}"

        # Update SDXL toggle state and trigger its text update
        self.sdxl_toggle.state = 'down' if self.use_sdxl else 'normal'
        self.on_sdxl_toggle(self.sdxl_toggle, self.sdxl_toggle.state) # Force text update

        # Update step label and progress bar max
        self.update_steps_label(None, self.steps_slider.value)

        # Attempt to preload model (if path is valid after loading cache)
        if self.model_path and os.path.exists(self.model_path):
            self.preload_model(0) # Call directly now, no need for another clock schedule
        elif self.model_path: # Path existed in cache but file doesn't
            self.model_label.text = f"Cached model not found: {os.path.basename(self.model_path)}"


    # --- load_settings_from_cache (Revised to set all state attributes) ---
    def load_settings_from_cache(self):
        """Load settings from cache file and set instance attributes."""
        print("Loading settings from cache...")
        # Use get() with defaults matching the initializations in build()
        self.model_path = self.current_cache.get("model_path", "")
        self.use_sdxl = self.current_cache.get("use_sdxl", False)
        self.custom_width = self.current_cache.get("custom_width")
        self.custom_height = self.current_cache.get("custom_height")

        # Determine selected_resolution based on custom settings
        if self.custom_width and self.custom_height:
            self.selected_resolution = None # Indicate custom is active
        else:
            # Get resolution from cache, validate, or use default
            cached_res = self.current_cache.get("resolution")
            valid_res_values = [r[1] for r in RESOLUTIONS]
            if cached_res in valid_res_values:
                self.selected_resolution = cached_res
            else:
                # If invalid/missing, use the default value
                self.selected_resolution = DEFAULT_RESOLUTION_VALUE
                if cached_res is not None: # Only warn if cache had an invalid value
                    print(f"Warning: Invalid resolution '{cached_res}' in cache, using default {self.selected_resolution}.")

        # Load mode, validate, or use default
        cached_mode = self.current_cache.get("mode", "Normal")
        if cached_mode in self.available_modes:
            self.selected_mode = cached_mode
        else:
            self.selected_mode = "Normal"
            if cached_mode is not None:
                 print(f"Warning: Invalid mode '{cached_mode}' in cache, using default 'Normal'.")

        # Load text inputs and steps/seed
        self.steps_value = self.current_cache.get("steps", STEPS_DEFAULT)
        self.prompt_input_text = self.current_cache.get("prompt", "")
        self.neg_prompt_input_text = self.current_cache.get("negative_prompt", "")
        self.seed_text = str(self.current_cache.get("seed", "1"))
        if not self.seed_text.strip(): self.seed_text = "1" # Ensure seed is never empty string internally

        print(f" Settings loaded: Res={self.selected_resolution}, Mode={self.selected_mode}, SDXL={self.use_sdxl}")


    # --- Other methods (preload_model, update_steps_label, on_sdxl_toggle, etc.) ---
    # (Keep the rest of the methods as in the previous version)
    def preload_model(self, dt):
         if not self.model_path or not os.path.exists(self.model_path):
              print("Cached model path invalid or missing, skipping preload.")
              return
         print("Preloading model from cache...")
         self.model_label.text = f"Loading: {os.path.basename(self.model_path)}"
         threading.Thread(target=self.load_model_thread, daemon=True).start()

    def update_steps_label(self, instance, value):
        steps = int(value)
        self.steps_value_label.text = str(steps)
        # Update internal state if needed, though slider value is source of truth
        self.steps_value = steps
        self.progress_bar.max = steps
        if self.generate_btn.disabled == False:
             self.progress_label.text = f"Progress: 0/{steps}"

    def on_sdxl_toggle(self, instance, value):
        new_sdxl_state = value == 'down'
        instance.text = "SDXL: ON" if new_sdxl_state else "SDXL: OFF"
        if new_sdxl_state != self.use_sdxl:
            self.use_sdxl = new_sdxl_state
            print(f"SDXL mode toggled: {self.use_sdxl}")
            if self.model_path and self.pipeline_manager.pipe is not None:
                 self.show_info("SDXL mode changed. Reloading model...")
                 Clock.schedule_once(self.reload_model_for_sdxl_change, 0.1)

    def reload_model_for_sdxl_change(self, dt):
         if self.model_path:
              self.model_label.text = f"Reloading: {os.path.basename(self.model_path)} (SDXL: {self.use_sdxl})"
              threading.Thread(target=self.load_model_thread, daemon=True).start()
         else:
              self.show_info("Select a model file first.")


    def set_resolution(self, res, text):
        self.selected_resolution = res
        self.custom_width = None
        self.custom_height = None
        self.resolution_button.text = text
        self.resolution_dropdown.dismiss()
        self.custom_res_label.text = ""
        print(f"Resolution set to: {res}x{res}")


    def set_generation_mode(self, mode_name):
        self.selected_mode = mode_name
        self.mode_button.text = f"Mode: {mode_name}"
        self.mode_dropdown.dismiss()
        print(f"Generation mode set to: {mode_name}")

    def show_custom_resolution_popup(self, instance):
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        current_w = self.custom_width or self.selected_resolution or DEFAULT_RESOLUTION_VALUE
        current_h = self.custom_height or self.selected_resolution or DEFAULT_RESOLUTION_VALUE
        content.add_widget(Label(text="Width (multiple of 8, >= 64):"))
        width_inp = TextInput(text=str(current_w), multiline=False, input_filter="int")
        content.add_widget(width_inp)
        content.add_widget(Label(text="Height (multiple of 8, >= 64):"))
        height_inp = TextInput(text=str(current_h), multiline=False, input_filter="int")
        content.add_widget(height_inp)
        ok_btn = Button(text="Set Custom Resolution", size_hint=(1, None), height=40)
        content.add_widget(ok_btn)
        popup = Popup(title="Custom Resolution", content=content, size_hint=(0.8, 0.6))
        def on_ok(btn_instance):
            try:
                w = int(width_inp.text)
                h = int(height_inp.text)
                if w < 64 or h < 64 or w % 8 != 0 or h % 8 != 0:
                    raise ValueError("Width/Height must be >= 64 and multiples of 8.")
                self.custom_width, self.custom_height = w, h
                self.selected_resolution = None
                self.resolution_button.text = "Custom Res"
                self.custom_res_label.text = f"Custom: {w}x{h}"
                popup.dismiss()
                print(f"Custom resolution set: {w}x{h}")
            except ValueError as e: self.show_error(str(e))
            except Exception as e: self.show_error(f"Invalid input: {e}")
        ok_btn.bind(on_press=on_ok)
        popup.open()

    def select_model_file(self, instance):
        start_path = os.path.dirname(self.model_path) if self.model_path and os.path.exists(os.path.dirname(self.model_path)) else os.path.expanduser("~")
        content = BoxLayout(orientation='vertical')
        fc = FileChooserListView(filters=["*.safetensors"], path=start_path, size_hint=(1, 0.9))
        select_button = Button(text="Select", size_hint=(1, 0.1))
        content.add_widget(fc)
        content.add_widget(select_button)
        popup = Popup(title="Select Model File (.safetensors)", content=content, size_hint=(0.9, 0.9))
        def on_select_press(btn_instance):
             if fc.selection:
                 self.on_model_selected(fc, fc.selection)
                 popup.dismiss()
             else:
                 self.show_error("No file selected.")
        select_button.bind(on_press=on_select_press)
        popup.open()


    def on_model_selected(self, instance, selection):
        if selection:
            new_model_path = selection[0]
            needs_reload = (
                new_model_path != self.model_path or
                self.pipeline_manager.pipe is None or
                self.pipeline_manager.use_sdxl != self.use_sdxl
            )
            if needs_reload:
                self.model_path = new_model_path
                self.model_label.text = f"Loading: {os.path.basename(self.model_path)}"
                self.show_info("Loading model...")
                threading.Thread(target=self.load_model_thread, daemon=True).start()
            else:
                 print("Selected model is already loaded and matches SDXL state.")
                 self.model_label.text = f"Loaded: {os.path.basename(self.model_path)}"


    def load_model_thread(self):
        success = self.pipeline_manager.load_model(self.model_path, self.use_sdxl)
        Clock.schedule_once(lambda dt: self.update_model_status(success), 0)

    def update_model_status(self, success):
        if success:
            self.model_label.text = f"Loaded: {os.path.basename(self.model_path)}"
            self.show_info("Model loaded successfully.")
            self.save_current_settings()
        else:
            self.show_error(f"Failed to load model: {os.path.basename(self.model_path)}")
            self.model_label.text = "Model Load Failed!"
            self.model_path = ""


    def start_generation(self, instance):
        if not self.model_path or self.pipeline_manager.pipe is None:
            self.show_error("Load a model first.")
            return
        # Read directly from UI widgets at the time of generation start
        prompt_text = self.prompt_input.text.strip()
        neg_prompt_text = self.neg_prompt_input.text.strip()
        seed_text = self.seed_input.text.strip()
        steps_val = int(self.steps_slider.value) # Read slider value directly

        if not prompt_text:
            self.show_error("Please enter a prompt.")
            return

        if self.custom_width and self.custom_height:
            width, height = self.custom_width, self.custom_height
        elif self.selected_resolution:
            width = height = self.selected_resolution
        else:
             self.show_error("Please select a resolution.")
             return

        try:
            if seed_text: int(seed_text)
            else: seed_text = ""
        except ValueError:
            self.show_error(f"Invalid seed value: '{seed_text}'. Using numbers only.")
            return

        self.generate_btn.disabled = True
        self.cancel_btn.disabled = False
        self.cancel_requested.clear()
        self.progress_bar.value = 0
        self.progress_label.text = f"Starting... (0/{steps_val})"
        self.info_label.text = "Generating..."
        self.image_display.texture = None

        # Update internal state just before saving/threading
        self.prompt_input_text = prompt_text
        self.neg_prompt_input_text = neg_prompt_text
        self.seed_text = seed_text if seed_text else "1" # Ensure internal seed state isn't empty
        self.steps_value = steps_val

        gen_params = {
            "prompt": prompt_text,
            "negative_prompt": neg_prompt_text,
            "width": width, "height": height, "steps": steps_val,
            "cfg_scale": CFG_SCALE, "seed": seed_text, # Pass original seed text (empty OK)
            "mode": self.selected_mode,
            "progress_callback": self.update_progress
        }

        self.save_current_settings() # Save the state we just updated

        self.generation_thread = threading.Thread(target=self.run_generation, args=(gen_params,), daemon=True)
        self.generation_thread.start()

    def save_current_settings(self):
        """Save current internal state to the cache file."""
        data = {
            "model_path": self.model_path,
            "use_sdxl": self.use_sdxl,
            "resolution": self.selected_resolution, # Will be None if custom
            "custom_width": self.custom_width,
            "custom_height": self.custom_height,
            "steps": self.steps_value,
            "seed": self.seed_text, # Saved internal state (default "1")
            "prompt": self.prompt_input_text,
            "negative_prompt": self.neg_prompt_input_text,
            "mode": self.selected_mode
        }
        save_cache(data)


    def run_generation(self, params):
        error_message_for_popup = None
        try:
            final_image, gen_time = self.pipeline_manager.generate_image(**params)

            if self.cancel_requested.is_set():
                 print("Generation cancelled by user.")
                 Clock.schedule_once(lambda dt: self.show_info("Generation Cancelled."), 0)
                 return

            if final_image:
                os.makedirs(OUTPUT_DIR, exist_ok=True)
                timestamp = int(time.time())
                actual_seed = params['seed'] if params['seed'] else 'random'
                out_filename = f"output_{timestamp}_{params['width']}x{params['height']}_{actual_seed}_{params['mode']}.png"
                out_path = os.path.join(OUTPUT_DIR, out_filename)
                final_image.save(out_path)
                print(f"Image saved to: {out_path}")
                Clock.schedule_once(lambda dt: self.update_ui_after_generation(final_image, gen_time, out_path), 0)
            else:
                 if not self.cancel_requested.is_set():
                      error_message_for_popup = "Generation finished but produced no image."

        except Exception as e:
            error_message_for_popup = f"Generation Error: {e}\n\n{traceback.format_exc()}"
            print(f"!!! Generation thread error: {e}")

        finally:
            if error_message_for_popup:
                 Clock.schedule_once(lambda dt, msg=error_message_for_popup: self._show_generation_error(msg), 0)
            Clock.schedule_once(lambda dt: self.reset_buttons(), 0)


    def _show_generation_error(self, message):
         self.show_error(message)


    def update_ui_after_generation(self, pil_image, gen_time, image_path):
        try:
            texture = self.pil_image_to_texture(pil_image)
            if texture:
                 self.image_display.texture = texture
            else:
                 self.image_display.texture = None
            self.info_label.text = f"Done! Time: {gen_time:.2f}s | Saved: {os.path.basename(image_path)}"
            print(f"UI Update: {get_memory_usage()}")
        except Exception as e:
            print(f"Error updating UI with image: {e}")
            self.show_error(f"UI Update Error: {e}")


    def update_progress(self, step, total_steps):
        Clock.schedule_once(lambda dt: self._update_progress_ui(step, total_steps), 0)

    def _update_progress_ui(self, step, total_steps):
        if total_steps > 0:
            self.progress_bar.max = total_steps
            self.progress_bar.value = min(step, total_steps)
            self.progress_label.text = f"Progress: {step}/{total_steps}"
        else:
             self.progress_bar.value = 0
             self.progress_label.text = "Progress: ..."


    def reset_buttons(self):
        self.generate_btn.disabled = False
        self.cancel_btn.disabled = True
        current_info = self.info_label.text
        if current_info.startswith(("Generating...", "Starting...", "Cancelling...")):
             self.info_label.text = "Ready."

    def cancel_generation(self, instance):
        if self.generation_thread and self.generation_thread.is_alive():
            print("Cancel requested...")
            self.cancel_requested.set()
            self.info_label.text = "Cancelling..."
            self.cancel_btn.disabled = True
        else:
            print("No active generation to cancel.")
            self.reset_buttons()


    def show_error(self, msg):
        print(f"ERROR: {msg}")
        content = BoxLayout(orientation='vertical', spacing=10, padding=10)
        scroll_label = ScrollView(size_hint=(1, 0.8))
        error_input = TextInput(text=str(msg), size_hint_y=None, readonly=True, background_color=(0.1,0.1,0.1,1), foreground_color=(1,0.7,0.7,1))
        error_input.bind(minimum_height=error_input.setter('height'))
        scroll_label.add_widget(error_input)
        content.add_widget(scroll_label)
        close_button = Button(text='Close', size_hint=(1, 0.15))
        content.add_widget(close_button)
        popup = Popup(title="Error", content=content, size_hint=(0.9, 0.6))
        close_button.bind(on_press=popup.dismiss)
        popup.open()

    def show_info(self, msg):
        print(f"INFO: {msg}")
        self.info_label.text = msg


    def show_debug_output(self, instance):
        content = BoxLayout(orientation="vertical", spacing=10, padding=10)
        debug_text_area = ScrollView(size_hint=(1, 0.9))
        debug_log = get_debug_output()
        debug_input = TextInput(text=debug_log, size_hint_y=None, readonly=True, font_size='10sp')
        debug_input.bind(minimum_height=debug_input.setter('height'))
        debug_text_area.add_widget(debug_input)
        close_btn = Button(text="Close", size_hint=(1, 0.1))
        content.add_widget(debug_text_area)
        content.add_widget(close_btn)
        popup = Popup(title="Debug Output Log", content=content, size_hint=(0.9, 0.9))
        close_btn.bind(on_press=popup.dismiss)
        popup.open()


    def pil_image_to_texture(self, pil_image):
        if pil_image is None:
            print("Warning: Attempted to convert None PIL image to texture.")
            return None
        try:
            pil_image = pil_image.convert("RGB")
            img_data = pil_image.tobytes()
            texture = Texture.create(size=pil_image.size, colorfmt="rgb")
            texture.blit_buffer(img_data, colorfmt="rgb", bufferfmt="ubyte")
            texture.flip_vertical()
            return texture
        except Exception as e:
            print(f"Texture conversion error: {e}")
            self.show_error(f"Texture conversion failed: {e}")
            return None

    def on_stop(self):
        print("Application stopping...")
        self.cancel_requested.set()
        if self.generation_thread and self.generation_thread.is_alive():
             print("Waiting briefly for generation thread...")
             self.generation_thread.join(timeout=1.0)
        print("Unloading model...")
        if hasattr(self, 'pipeline_manager') and self.pipeline_manager:
            self.pipeline_manager.unload_model()
        memory_cleanup()
        print("Cleanup complete on stop.")