import soundfile as sf
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# Replace with your downloaded file path
audio_file = 'final_project/data/MARS-20240101T000000Z-16kHz.wav'

# Get info using soundfile
info = sf.info(audio_file)
print(info)

# Load first N seconds for fast inspection (e.g., 60 seconds)
y, sr = librosa.load(audio_file, sr=None, duration=60)
print(f"Loaded audio duration (sec): {len(y)/sr:.2f} | Sampling Rate: {sr}")

# Plot waveform
plt.figure(figsize=(14, 4))
librosa.display.waveshow(y, sr=sr)
plt.title('Waveform (first 60 seconds)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.tight_layout()
plt.show()

# Plot Mel spectrogram
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
S_dB = librosa.power_to_db(S, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
plt.title('Mel-frequency Spectrogram')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()

segment_length = 10  # seconds
hop_length = 5       # seconds (overlap)
save_dir = "final_project/data/spectrogram_chunks"
os.makedirs(save_dir, exist_ok=True)

total_duration = len(y) / sr
segment_samples = int(segment_length * sr)
hop_samples = int(hop_length * sr)

i = 0
for start in range(0, len(y) - segment_samples + 1, hop_samples):
    chunk = y[start: start + segment_samples]
    S = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)
    # Normalize and convert to image
    S_img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min())
    S_img = (S_img * 255).astype(np.uint8)
    im = Image.fromarray(S_img)
    im.save(os.path.join(save_dir, f'spect_{i:04d}.png'))
    i += 1

print(f"Saved {i} spectrogram images to {save_dir}")