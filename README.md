# MBARI MAE-AST: Marine Audio Self-Supervised Learning

Self-supervised pretraining of Audio Spectrogram Transformers on MBARI marine acoustic data.

## 🚀 Quick Start

1. **Setup Environment**

   Due to different package requirements, there are two separate environments to run the two models used in the project. The mae-ast model works with `Python 3.9` with the following requirement file:
   ```bash
   pip install -r requirements.txt
   ```
   The downstream model works with `Python 3.11` with the following requirement file:
   ```bash
   pip install -r requirements_downstream.txt
   ```
   You need to install Pytorch version compatible with your GPU driver. In particular, the downstream model requires `Pytorch version>=2.6`.

2. **Data**

   MBARI dataset can be downloaded from [here](https://docs.mbari.org/pacific-sound/quickstart/); you might want to install `aws-cli` from [here](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).

   Downstream Watkins dataset can be downloaded using `notebook/downstream/download_watkins.py`.


3. **Audio Clips & tsv Generator**

   You can use `make_clips_flexible.py` to automatically generate clips from the downloaded audio files from MBARI dataset. In case you want to only generate tsv file from the audio clips, you could use `generate_tsv_manifests.py`. For example,
   ```bash
   python scripts/data_processing/generate_tsv_manifests.py --input_folder audio_chunks-MARS-20171030T000000Z-10secs
   ```

4. **Experiments**

   As for MBARI dataset with mae-ast, you could use `mae_ast_batch_sweep.sh` or similarly named python files to run through the parameters of interest.

   As for WAtkins dataset, refer to the notebook files for experiments.



## ⚠️ Important Code Locations and Usage
Locations:
```
scripts/data_processing
data
data/manifests          # related tsv folders
config
outputs
```


