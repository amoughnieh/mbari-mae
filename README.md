# MBARI MAE-AST: Marine Audio Self-Supervised Learning

Self-supervised pretraining of Audio Spectrogram Transformers on MBARI marine acoustic data.

### 🚧 In Progress / TODO
- **TSV manifest generator** - should be added.
- **Training pipeline**

### Onboarding ###
- Edit requirements.txt for your CUDA version.
- Add a .gitkeep file into each data folder; only the folder structure is pushed, not the data.
- Feel free to edit .gitignore as needed.

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

## 🚀 Quick Start

1. **Setup Environment**
   ```bash
   # Edit requirements.txt for your CUDA version (cu121, cu118, etc.)
   pip install -r requirements.txt
   ```

---