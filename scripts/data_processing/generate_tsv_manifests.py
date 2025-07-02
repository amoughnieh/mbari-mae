#!/usr/bin/env python3
"""
Generate train/valid TSV manifests for audio chunks.

Usage:
    python generate_tsv_manifests.py --input_folder audio_chunks-MARS-20171030T000000Z-10secs --train_ratio 0.9  # train_ratio is optional, default by 0.95

Creates manifests in: data/manifests/{input_folder_name}
"""

import argparse
import sys
from pathlib import Path
import random
import soundfile as sf


def generate_tsv_manifests(input_folder: str, train_ratio: float = 0.95, seed: int = 42):
    """
    Generate train/valid TSV manifests for an audio chunks folder using temporal splitting.
    
    Args:
        input_folder: 
        train_ratio: Fraction for training (remainder goes to validation)
        seed: Unused for now.
    
    Note:
        Uses temporal splitting to avoid data leakage from overlapping audio segments.
        Training set contains earlier files, validation set contains later files.
    """
    
    # Setup paths.
    project_root = Path(__file__).parent.parent.parent  # Go up to project root.
    audio_dir = project_root / "data" / input_folder
    manifest_dir = project_root / "data" / "manifests" / input_folder
    
    # Validation.
    if not audio_dir.exists():
        print(f"ERROR: Audio directory not found: {audio_dir}")
        print(f"   Available folders in data/:")
        data_dir = project_root / "data"
        if data_dir.exists():
            for item in data_dir.iterdir():
                if item.is_dir():
                    print(f"   - {item.name}")
        sys.exit(1)
    
    # Find audio files.
    wav_files = sorted(audio_dir.glob("*.wav"))
    if not wav_files:
        print(f"ERROR: No .wav files found in {audio_dir}")
        sys.exit(1)
    
    print(f"Found {len(wav_files)} audio files")
    
    # Use temporal splitting instead of random splitting.
    #
    # Example with overlap:         
    #   Split: chunk_001-142 → train, chunk_143-152 → valid.

    wav_files = sorted(wav_files)  # Ensure consistent temporal ordering.
    
    # Calculate splits using sequential splitting.
    n_total = len(wav_files)
    n_train = int(n_total * train_ratio)
    n_valid = n_total - n_train
    
    splits = {
        "train": wav_files[:n_train],           # First N files (earlier in time).
        "valid": wav_files[n_train:]            # Last M files (later in time).
    }
    
    print(f"Temporal Split: Train={n_train} ({train_ratio:.1%}), Valid={n_valid} ({(1-train_ratio):.1%})")
    print(f"   → Train: files 1-{n_train} (earlier), Valid: files {n_train+1}-{n_total} (later)")
    
    # Create manifest directory
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate TSV files
    for split_name, files in splits.items():
        tsv_path = manifest_dir / f"{split_name}.tsv"
        
        with tsv_path.open("w") as f:
            # Header: absolute path to audio root directory (required by fairseq).
            f.write(f"{audio_dir.resolve()}\n")
            
            # Write each file with its sample count.
            for wav_file in files:
                try:
                    frames = sf.info(wav_file).frames
                    f.write(f"{wav_file.name}\t{frames}\n")
                except Exception as e:
                    print(f"WARNING: Skipping {wav_file.name} - {e}")
                    continue
        
        print(f"✅ Created: {tsv_path} ({len(files)} files)")
    
    print(f"Manifests saved to: {manifest_dir}")
    return manifest_dir


def main():
    parser = argparse.ArgumentParser(
        description="Generate train/valid TSV manifests for audio chunks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_tsv_manifests.py --input_folder audio_chunks-MARS-20171030T000000Z-10secs
        """
    )
    
    parser.add_argument(
        "--input_folder", 
        type=str, 
        required=True,
        help="Name of audio folder within data/ directory"
    )
    
    parser.add_argument(
        "--train_ratio", 
        type=float, 
        default=0.95,
        help="Fraction of data for training (default: 0.95 for 95/5 split)"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed (currently unused, kept for future extensibility)"
    )
    
    args = parser.parse_args()
    
    # Validate train_ratio.
    if not 0.5 <= args.train_ratio <= 0.99:
        print(f"ERROR: train_ratio must be between 0.5 and 0.99, got {args.train_ratio}")
        sys.exit(1)
    
    # Generate manifests.
    manifest_dir = generate_tsv_manifests(
        input_folder=args.input_folder,
        train_ratio=args.train_ratio, 
        seed=args.seed
    )


if __name__ == "__main__":
    main() 