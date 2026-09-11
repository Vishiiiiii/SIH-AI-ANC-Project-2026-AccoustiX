import soundfile as sf, pathlib

rates = {}
for p in pathlib.Path("data/raw/noise").rglob("*"):
    if p.suffix.lower() in (".wav", ".flac"):
        info = sf.info(str(p))
        rates[info.samplerate] = rates.get(info.samplerate, 0) + 1
print(rates)
