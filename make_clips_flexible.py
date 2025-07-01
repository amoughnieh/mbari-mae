#!/usr/bin/env python3
import argparse
from pathlib import Path
import random

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm


def make_clips(input_dir: Path, output_dir: Path, segment_duration: float,
               train_tsv_path: Path, valid_tsv_path: Path, valid_ratio: float,
               seed: int, overlap_duration: float = 0.0, silence_threshold: float = 0.01,
               n_mels: int = 128, mel_threshold: float = -10.0):
    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    # stop generating after this many clips total
    max_clips = 200000
    stopped = False

    # Iterate over all .wav files
    wav_paths = sorted(input_dir.glob("*.wav"))
    for wav_path in tqdm(wav_paths, desc="Files", unit="file"):
        data, sr = sf.read(str(wav_path))
        total_samples = data.shape[0]
        seg_samples = int(segment_duration * sr)

        clips_to_process = []  # A list to hold clips generated from this one file

        if total_samples < seg_samples:
            # --- CASE 1: File is SHORTER than segment_duration ---
            # Treat the whole file as a single clip without segmentation.
            clips_to_process.append((data, f"{wav_path.stem}.wav"))
        else:
            # --- CASE 2: File is LONG ENOUGH for segmentation ---
            # Apply the original segmentation logic.
            overlap_samples = int(overlap_duration * sr)
            hop_samples = seg_samples - overlap_samples if seg_samples > overlap_samples else seg_samples
            n_segs = (total_samples + hop_samples - 1) // hop_samples

            for i in range(n_segs):
                start = i * hop_samples
                end = start + seg_samples
                if end <= total_samples:
                    clip_data = data[start:end]
                else:
                    clip_data = data[start:total_samples]
                    # Pad the final clip to ensure uniform length for segmented files
                    clip_data = np.pad(clip_data, (0, seg_samples - clip_data.shape[0]), mode="constant")

                clips_to_process.append((clip_data, f"{wav_path.stem}_seg{i:04d}.wav"))

        # --- Now, process all clips generated from the file (either 1 or many) ---
        for clip, out_name in clips_to_process:
            # silence filter via RMS threshold
            rms = np.sqrt(np.mean(clip ** 2))
            if rms < silence_threshold:
                continue
            # mel-spectrogram analysis
            mel_spec = librosa.feature.melspectrogram(y=clip, sr=sr, n_mels=n_mels)
            log_mel = librosa.power_to_db(mel_spec, ref=np.max)
            # normalize spectrogram
            log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
            # drop clips without sufficient energy in mel bands
            if log_mel.max() < mel_threshold:
                continue

            # Save the good clip
            out_path = output_dir / out_name
            sf.write(str(out_path), clip, sr)

            # Record path and sample length
            records.append((out_path, clip.shape[0]))
            # stop if reached limit
            if len(records) >= max_clips:
                stopped = True
                break
        if stopped:
            break

    # Shuffle and split into training and validation sets
    random.shuffle(records)
    n_valid = int(len(records) * valid_ratio)
    valid_records = records[:n_valid]
    train_records = records[n_valid:]

    # Write train.tsv
    with train_tsv_path.open("w") as f_train:
        # first line: absolute audio root directory (so Fairseq can find files under Hydra)
        f_train.write(f"{output_dir.resolve()}\n")
        for path, length in train_records:
            # manifest entries should be relative to audio root
            rel_path = path.relative_to(output_dir)
            f_train.write(f"{rel_path}\t{length}\n")

    # Write valid.tsv
    with valid_tsv_path.open("w") as f_valid:
        # first line: absolute audio root directory
        f_valid.write(f"{output_dir.resolve()}\n")
        for path, length in valid_records:
            rel_path = path.relative_to(output_dir)
            f_valid.write(f"{rel_path}\t{length}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split WAVs into fixed-length clips and generate train/valid TSVs with sample counts."
    )
    parser.add_argument(
        "--input_dir", type=Path, required=True,
        help="Directory containing original .wav files"
    )
    parser.add_argument(
        "--output_dir", type=Path, required=True,
        help="Where to write 5s clips"
    )
    parser.add_argument(
        "--segment_duration", type=float, default=10.0,
        help="Clip length in seconds"
    )
    parser.add_argument(
        "--train_tsv_path", type=Path, required=True,
        help="Path to output train.tsv"
    )
    parser.add_argument(
        "--valid_tsv_path", type=Path, required=True,
        help="Path to output valid.tsv"
    )
    parser.add_argument(
        "--valid_ratio", type=float, default=0.1,
        help="Fraction of clips for validation"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--overlap_duration", type=float, default=0.0,
        help="Overlap length in seconds between consecutive clips"
    )
    parser.add_argument(
        "--silence_threshold", type=float, default=0.01,
        help="Minimum RMS amplitude for a clip to be kept"
    )
    parser.add_argument(
        "--n_mels", type=int, default=128,
        help="Number of mel bands for spectrogram"
    )
    parser.add_argument(
        "--mel_threshold", type=float, default=-40.0,
        help="Minimum log-mel dB max value to keep clip"
    )
    args = parser.parse_args()
    make_clips(
        args.input_dir,
        args.output_dir,
        args.segment_duration,
        args.train_tsv_path,
        args.valid_tsv_path,
        args.valid_ratio,
        args.seed,
        args.overlap_duration,
        args.silence_threshold,
        args.n_mels,
        args.mel_threshold,
    )
