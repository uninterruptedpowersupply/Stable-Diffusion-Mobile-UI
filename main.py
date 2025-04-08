# main.py
import os
import sys
import platform # Import platform here

# --- Early Setup ---
# Environment variables for Torch are now set in memory_utils.py
print(f"PYTORCH_CUDA_ALLOC_CONF: {os.environ.get('PYTORCH_CUDA_ALLOC_CONF', 'Not Set (Using Default)')}")


# --- Import Kivy and Torch ---
try:
    import torch
    from kivy.config import Config
    # Config.set('graphics', 'multisamples', '0') # Disable if causing issues

    from kivy.core.window import Window
    from ui_interface import SDAppInterface # Your App class
    from memory_utils import configure_system, memory_cleanup, get_memory_usage
    from debug_stream import get_debug_output
except ImportError as e:
    print(f"Fatal Error: Missing required library - {e}")
    print("Please install dependencies from requirements.txt")
    # Simple graphical error if Kivy is available
    try:
        from kivy.app import App
        from kivy.uix.label import Label
        from kivy.uix.popup import Popup
        from kivy.uix.boxlayout import BoxLayout
        class ErrorApp(App):
            def build(self):
                layout = BoxLayout(orientation='vertical')
                layout.add_widget(Label(text=f'Failed to import modules:\n{e}\nPlease check installation.'))
                popup = Popup(title='Import Error', content=layout, size_hint=(0.8, 0.4))
                popup.open()
                return Label() # Dummy widget
        ErrorApp().run()
    except Exception:
        pass
    sys.exit(1)


# --- System Configuration ---
print("Starting Application...")
configure_system() # Apply Torch/CUDA backend settings

# --- Window Size ---
try:
    if "ANDROID_ARGUMENT" in os.environ:
         print("Detected Android platform, adjusting window size.")
         Window.size = (400, 700)
    elif platform.system() in ["Windows", "Darwin", "Linux"]:
         print("Detected Desktop platform.")
         Window.size = (600, 850) # Slightly taller for better layout
    else:
         print("Unknown platform, using default size.")
         Window.size = (600, 850)
except Exception as e:
     print(f"Platform detection/window size error: {e}. Using default.")
     Window.size = (600, 850)


# --- Run the App ---
if __name__ == "__main__":
    app_instance = None
    exit_code = 0
    try:
        print("Initializing Kivy UI...")
        print(f"Initial Memory: {get_memory_usage()}")
        app_instance = SDAppInterface()
        app_instance.run()
    except KeyboardInterrupt:
         print("\nKeyboardInterrupt received. Stopping application.")
         # App stop event should handle cleanup via on_stop
    except Exception as e:
        import traceback
        print("\n--- UNHANDLED APPLICATION ERROR ---")
        print(traceback.format_exc())
        print("-----------------------------------\n")
        exit_code = 1 # Indicate error on exit
        try:
            log_content = f"--- Crash Log ---\n{traceback.format_exc()}\n\n--- Debug Stream ---\n{get_debug_output()}"
            with open("crash_log.txt", "w", encoding='utf-8') as f:
                f.write(log_content)
            print("Crash log saved to crash_log.txt")
        except Exception as log_e:
            print(f"Could not save crash log: {log_e}")

    finally:
        print("Application exiting process initiated.")
        # on_stop method in the App class handles the primary cleanup.
        print(f"Exiting with code {exit_code}.")
        sys.exit(exit_code)