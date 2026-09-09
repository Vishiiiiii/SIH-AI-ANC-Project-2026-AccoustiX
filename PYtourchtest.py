import torch
import torchaudio
import librosa
import soundfile as sf
import sounddevice as sd
import numpy as np
import scipy
import matplotlib
import onnx
import onnxruntime

print("=== SETUP STATUS ===")
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only'}")
print(f"Torchaudio: {torchaudio.__version__}")
print(f"Librosa: {librosa.__version__}")
print(f"Soundfile: {sf.__version__}")
print(f"Sounddevice: {sd.__version__}")
print(f"NumPy: {np.__version__}")
print(f"SciPy: {scipy.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"ONNX: {onnx.__version__}")
print(f"ONNX Runtime: {onnxruntime.__version__}")
print("=====================")