import wave
import struct
import math
import os

sample_rate = 16000
duration = 2.0  # 2 seconds (needs padding to 5 seconds by preprocess.py)
freq = 440.0
num_samples = int(sample_rate * duration)

os.makedirs('data/raw/real', exist_ok=True)
filename = 'data/raw/real/dummy.wav'

with wave.open(filename, 'w') as wav_file:
    wav_file.setnchannels(1)
    wav_file.setsampwidth(2)
    wav_file.setframerate(sample_rate)
    
    for i in range(num_samples):
        value = int(32767.0 * math.sin(2.0 * math.pi * freq * i / sample_rate))
        data = struct.pack('<h', value)
        wav_file.writeframesraw(data)

print(f"Created {filename}")
