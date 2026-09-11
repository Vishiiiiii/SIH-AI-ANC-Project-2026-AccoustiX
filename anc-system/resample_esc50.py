import soundfile as sf
import librosa
import pathlib

esc50_dir = pathlib.Path("data/raw/noise/esc50/ESC-50-master/audio")
target_sr = 16000

wav_files = list(esc50_dir.glob("*.wav"))
print(f"Found {len(wav_files)} files to check.")

converted = 0
for f in wav_files:
    data, sr = sf.read(f)
    if sr == target_sr:
        continue
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono while we're at it
    resampled = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
    sf.write(f, resampled, target_sr)
    converted += 1

print(f"Resampled {converted}/{len(wav_files)} files to {target_sr}Hz.")