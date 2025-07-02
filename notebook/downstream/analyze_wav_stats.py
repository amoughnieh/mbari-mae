import os
import torchaudio
import numpy as np
from pathlib import Path
from collections import defaultdict

def analyze_wav_files(root_dir):
    all_durations = []
    species_durations = defaultdict(list)
    
    for species_dir in Path(root_dir).iterdir():
        if not species_dir.is_dir():
            continue
            
        species_name = species_dir.name
        print(f"\nAnalyzing {species_name}...")
        
        for wav_file in species_dir.glob("*.wav"):
            try:
                waveform, sample_rate = torchaudio.load(wav_file)
                duration_seconds = waveform.shape[1] / sample_rate
                
                all_durations.append(duration_seconds)
                species_durations[species_name].append(duration_seconds)
                
            except Exception as e:
                print(f"Error processing {wav_file}: {e}")

    print("\n=== Overall Statistics ===")
    print(f"Total number of files: {len(all_durations)}")
    print(f"Minimum duration: {min(all_durations):.2f} seconds")
    print(f"Maximum duration: {max(all_durations):.2f} seconds")
    print(f"Median duration: {np.median(all_durations):.2f} seconds")
    print(f"Mean duration: {np.mean(all_durations):.2f} seconds")
    
    print("\n=== Per-Species Statistics ===")
    for species, durations in species_durations.items():
        print(f"\n{species}:")
        print(f"  Number of files: {len(durations)}")
        print(f"  Minimum duration: {min(durations):.2f} seconds")
        print(f"  Maximum duration: {max(durations):.2f} seconds")
        print(f"  Median duration: {np.median(durations):.2f} seconds")
        print(f"  Mean duration: {np.mean(durations):.2f} seconds")

if __name__ == "__main__":
    watkins_dir = "data/watkins"
    analyze_wav_files(watkins_dir) 