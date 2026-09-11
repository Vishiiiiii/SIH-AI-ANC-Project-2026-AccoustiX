import soundfile as sf
import librosa
import pathlib
import subprocess

FFMPEG_PATH = r"C:\ffmpeg\ffmpeg-9.0.1-essentials_build\bin\ffmpeg.exe"  # <-- paste your path, include ffmpeg.exe at the end

freenoise_dir = pathlib.Path("data/raw/noise/freenoise")
target_sr = 16000

wav_flac_files = list(freenoise_dir.rglob("*.wav")) + list(freenoise_dir.rglob("*.flac"))
mp3_files = list(freenoise_dir.rglob("*.mp3"))

print(f"Found {len(wav_flac_files)} wav/flac files and {len(mp3_files)} mp3 files.")

converted = 0
failed = []
for f in wav_flac_files:
    try:
        data, sr = sf.read(f)
    except Exception as e:
        failed.append((f, str(e)))
        continue
    if sr == target_sr and data.ndim == 1:
        continue
    if data.ndim > 1:
        data = data.mean(axis=1)
    resampled = librosa.resample(data.astype("float32"), orig_sr=sr, target_sr=target_sr)
    sf.write(f, resampled, target_sr)
    converted += 1

print(f"Resampled {converted}/{len(wav_flac_files)} wav/flac files to {target_sr}Hz mono.")
if failed:
    print(f"\n{len(failed)} files failed to open:")
    for f, err in failed:
        print(f"  {f.name}: {err}")

mp3_converted = 0
mp3_failed = []
for f in mp3_files:
    out_path = f.with_suffix(".wav")
    result = subprocess.run(
        [FFMPEG_PATH, "-y", "-i", str(f), "-ar", str(target_sr), "-ac", "1", str(out_path)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        mp3_failed.append((f, result.stderr[-300:]))
        continue
    f.unlink()
    mp3_converted += 1

print(f"\nConverted {mp3_converted}/{len(mp3_files)} mp3 files to wav @ {target_sr}Hz mono.")
if mp3_failed:
    print(f"{len(mp3_failed)} mp3 conversions failed:")
    for f, err in mp3_failed:
        print(f"  {f.name}: {err}")