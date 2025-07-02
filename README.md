# MBARI MAE-AST: Marine Audio Self-Supervised Learning

Self-supervised pretraining of Audio Spectrogram Transformers on MBARI marine acoustic data.

# ❗❗❗Please Read ❗❗❗
- Every uploaded **notebook** file should print out all the results and work within the current project structure.
- Before pushing upstream, notify in Microsoft teams, wait for a minute, and then push (naive but don't want to burden everyone anymore).
- When adding a data folder, create a file ```.gitkeep``` inside the folder, which will prevent uploading the files inside but retain the folder structure.
- Default venv name is ```venv``` which is included in ```.gitignore```; add yours if you want to use a different name.

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   # Edit requirements.txt for your CUDA version (cu121, cu118, etc.)
   pip install -r requirements.txt
   ```

2. **Generate tsv Files If Necessary**
   ```bash
   python scripts/data_processing/generate_tsv_manifests.py --input_folder audio_chunks-MARS-20171030T000000Z-10secs
   ---

3. For 2khz data, please have a look at ```mbari_10s_2khz.yaml``` and ```train_10s_simple.py```.

## ⚠️ Important Code Locations and Usage
Locations:
```
scripts/data_processing
data
data/manifests          # related tsv folders
config
outputs
```
Runs:
```
python scripts/data_processing/generate_tsv_manifests.py --input_folder audio_chunks-MARS-20171030T000000Z-10secs
```
You can check the relevant optional parameters in the codes.

## 📁 Experiment Structure

Each experiment should be organized as (proposed version, feel free to edit and discuss):
```
experiments/exp_XXX_description/
├── config.yaml         # Exact configuration used
├── checkpoints/        # Model checkpoints (.pt files)
├── logs/               # Training logs (tensorboard, text logs)
├── results/            # Metrics and evaluations (.json, .txt)
├── figures/            # Plots and visualizations (.png, .pdf)
└── README.md           # Experiment notes and findings
```

### 🚧 In Progress / TODO
- **TSV manifest generator** - should be added.
- **Training pipeline**

### 💡 Other Stuff ###
- Edit requirements.txt for your CUDA version.
- Feel free to edit .gitignore as needed.
