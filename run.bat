@echo off
call sdm_env\Scripts\activate.bat
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -U xformers --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python main.py