#!/usr/bin/env python3
"""
Create train/valid/test TSV manifests for every chunk sub-folder under
final_project/data/audio_chunks-*. Call once; it will generate:

data/tsv_10s, data/tsv_15s, data/tsv_30s
  ├── train.tsv
  ├── valid.tsv
  └── test.tsv
"""

from pathlib import Path
import soundfile as sf, random, argparse, sys

def build_one(chunk_dir: Path, out_dir: Path, valid_ratio=0.02, seed=42):
    wavs = sorted(chunk_dir.glob("*.wav"))
    if not wavs:
        print(f"[WARN] no wav files in {chunk_dir}", file=sys.stderr); return
    random.seed(seed); random.shuffle(wavs)

    n = len(wavs)
    valid_n = int(n * valid_ratio)
    test_n  = valid_n
    splits = {
        "train": wavs[: n - valid_n - test_n],
        "valid": wavs[n - valid_n - test_n : n - test_n],
        "test" : wavs[n - test_n :],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for split, files in splits.items():
        tsv = out_dir / f"{split}.tsv"
        with tsv.open("w") as f:
            f.write(str(chunk_dir.resolve()) + "\n")   # header = audio root
            for wav in files:
                frames = sf.info(wav).frames
                f.write(f"{wav.name}\t{frames}\n")
        print(f"{tsv}: {len(files)} clips")

def main():
    root = Path("/home/incantator/Documents/gt-dl-cs7643/final_project/data/data_chunks")
    mapping = {
        "audio_chunks-MARS-20171030T000000Z-10secs": "tsv_10s",
        "audio_chunks-MARS-20171030T000000Z-15secs": "tsv_15s",
        "audio_chunks-MARS-20171030T000000Z-30secs": "tsv_30s",
    }
    for sub, tsv_name in mapping.items():
        build_one(root / sub, root / tsv_name)

if __name__ == "__main__":
    main()
