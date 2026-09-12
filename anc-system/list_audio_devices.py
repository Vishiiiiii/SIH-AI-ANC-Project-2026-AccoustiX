"""
Lists audio host APIs and devices, with their default latency figures.
Run this BEFORE touching realtime_demo.py -- it tells you which device
index/host API to actually target for low latency on your machine.

Usage:
    python list_audio_devices.py
"""
import sounddevice as sd

print("=== Host APIs ===")
for i, api in enumerate(sd.query_hostapis()):
    print(f"[{i}] {api['name']}  (default input: {api['default_input_device']}, "
          f"default output: {api['default_output_device']})")

print("\n=== Devices ===")
for i, dev in enumerate(sd.query_devices()):
    io = []
    if dev["max_input_channels"] > 0:
        io.append(f"in={dev['max_input_channels']}ch")
    if dev["max_output_channels"] > 0:
        io.append(f"out={dev['max_output_channels']}ch")
    hostapi_name = sd.query_hostapis()[dev["hostapi"]]["name"]
    print(f"[{i}] {dev['name']!r:45s} ({hostapi_name:10s}) {' '.join(io):15s} "
          f"default_low_input_latency={dev['default_low_input_latency']*1000:.1f}ms "
          f"default_low_output_latency={dev['default_low_output_latency']*1000:.1f}ms")

print("\n=== Currently configured defaults ===")
print(sd.default.device)

print("\nOn Windows, look for a device whose host API is 'Windows WASAPI' rather "
      "than 'MME' or 'Windows DirectSound' -- WASAPI typically gives sub-20ms "
      "default_low_*_latency vs 100ms+ on MME. If your INMP441 setup shows under "
      "a WASAPI host API, note its device index; you'll pass that as `device=` to "
      "sd.Stream(...) in realtime_demo.py.")
