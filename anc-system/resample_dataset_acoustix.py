import soundfile as sf
import librosa
import pathlib

dataset_acoustix_dir = pathlib.Path("data/raw/noise/dataset_acoustix")
target_sr = 16000

wav_files = list(dataset_acoustix_dir.rglob("*.wav"))
print(f"Found {len(wav_files)} files to check under {dataset_acoustix_dir}.")

converted = 0
failed = []
for f in wav_files:
    try:
        data, sr = sf.read(f)
    except Exception as e:
        failed.append((f, str(e)))
        continue
    if sr == target_sr and data.ndim == 1:
        continue
    if data.ndim > 1:
        data = data.mean(axis=1)  # downmix to mono while we're at it
    resampled = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
    sf.write(f, resampled, target_sr)
    converted += 1

print(f"Resampled {converted}/{len(wav_files)} files to {target_sr}Hz mono.")
if failed:
    print(f"\n{len(failed)} files failed to open:")
    for f, err in failed:
        print(f"  {f}: {err}")
