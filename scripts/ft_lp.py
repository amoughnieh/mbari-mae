#%%
# Standard library
import datetime
import json
import os
from collections import Counter
import random


# Numerical computing
import numpy as np
import pandas as pd

# PyTorch
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader
import torchaudio.transforms as T
import torchaudio.functional as F

# Hugging Face Transformers
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

# Datasets
from datasets import load_dataset

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.model_summary import ModelSummary

# Metrics
from sklearn.utils.class_weight import compute_class_weight
from torchmetrics.classification import (
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassRecall,
)

# Visualization
import matplotlib.pyplot as plt

from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    ViTFeatureExtractor,
    ViTModel
)

# MAE-AST Library
from s3prl.nn.upstream import S3PRLUpstream

# Fix the length of the input audio to the same length
def _match_length_force(self, xs, target_max_len):
    xs_max_len = xs.size(1)
    if xs_max_len > target_max_len:
        xs = xs[:, :target_max_len, :]
    elif xs_max_len < target_max_len:
        pad_len = target_max_len - xs_max_len
        xs = torch.cat(
            (xs, xs[:, -1:, :].repeat(1, pad_len, 1)),
            dim=1
        )
    return xs

S3PRLUpstream._match_length = _match_length_force
#%%
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    pl.seed_everything(seed, workers=True)

#%%
# loading the dataset
data_dir = os.path.join("data", "watkins")
annotations_file_train = os.path.join(data_dir, "annotations.train.csv")
annotations_file_valid = os.path.join(data_dir, "annotations.valid.csv")
annotations_file_test = os.path.join(data_dir, "annotations.test.csv")

ds = load_dataset(
    "csv",
    data_files={"train": annotations_file_train,
                "validation": annotations_file_valid,
                "test": annotations_file_test},
)

for split_name in ["train", "validation", "test"]:
    split_dataset = ds[split_name]
    labels = split_dataset["label"]
    total = len(labels)
    counts = Counter(labels)

    print(f"{split_name.capitalize()} dataset: {total} examples, {len(counts)} classes")
    if "label" in split_dataset.features and hasattr(split_dataset.features["label"], "names"):
        class_names = split_dataset.features["label"].names
        for idx, name in enumerate(class_names):
            print(f"  {idx} ({name}): {counts.get(name, 0)}")
    else:
        for label, count in counts.items():
            print(f"  {label}: {count}")
#%%
# class weights calculation
train_labels = ds["train"]["label"]
unique_labels = sorted(set(train_labels))
label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
y_train = [label_to_int[lbl] for lbl in train_labels]

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(len(unique_labels)),
    y=y_train
)

num_classes = len(class_weights)
#%%
# model definition
class WMMDClassifier(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        backbone_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 0.05,
        max_epochs = 100,
        backbone: str = "facebook/wav2vec2-base",
        ckpt_path: str = "",
        finetune: bool = False,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters()

        if backbone == "facebook/wav2vec2-base":
            self.backbone     = Wav2Vec2Model.from_pretrained(backbone)
            self.embedding_dim = self.backbone.config.hidden_size

        elif backbone == "patrickvonplaten/tiny-wav2vec2-no-tokenizer":
            self.backbone     = Wav2Vec2Model.from_pretrained(backbone)
            self.embedding_dim = self.backbone.config.hidden_size

        elif backbone == "vit-imagenet":

            self.backbone = ViTModel.from_pretrained(
                "google/vit-base-patch16-224-in21k",
                add_pooling_layer=False
            )
            self.embedding_dim = self.backbone.config.hidden_size

        elif backbone.lower() == "mae-ast":
            #up_kwargs = {"name": "mae_ast_patch"}
            #s3 = S3PRLUpstream(**up_kwargs)

            # Check for the local checkpoint path first.
            if not ckpt_path:
                raise ValueError("For 'mae-ast' backbone, a local ckpt_path must be provided.")

            from s3prl.upstream.mae_ast.expert import UpstreamExpert
            self.backbone = UpstreamExpert(ckpt=ckpt_path)
            self.embedding_dim = 6144

            enc = self.backbone.model.encoder #s3.upstream.model.encoder
            enc.layers = nn.ModuleList(list(enc.layers))
            self.backbone.dec_sine_pos_embed = None
            self.backbone.decoder = None
            self.backbone.final_proj_reconstruction = None
            self.backbone.final_proj_classification  = None
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        try:
            self.backbone.gradient_checkpointing_enable()
        except Exception:
            pass

        for param in self.backbone.parameters():
            param.requires_grad = finetune

        if finetune:
            self.classifier = nn.Sequential(
                #nn.Dropout(0.3),
                #nn.Linear(self.embedding_dim, 1024),
                #nn.ReLU(inplace=True),
                nn.LayerNorm(self.embedding_dim),
                nn.Dropout(0.5),
                nn.Linear(self.embedding_dim, num_classes),
            )

        else:
            self.classifier = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Dropout(0.5),
                nn.Linear(self.embedding_dim, num_classes),
            )

        if class_weights is not None:
            cw = torch.tensor(class_weights, dtype=torch.float)
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        else:
            self.criterion = nn.CrossEntropyLoss()

        metrics_kwargs = dict(num_classes=num_classes, average='macro')
        self.train_precision = MulticlassPrecision(**metrics_kwargs)
        self.train_recall = MulticlassRecall(**metrics_kwargs)
        self.train_f1 = MulticlassF1Score(**metrics_kwargs)
        self.val_precision = MulticlassPrecision(**metrics_kwargs)
        self.val_recall = MulticlassRecall(**metrics_kwargs)
        self.val_f1 = MulticlassF1Score(**metrics_kwargs)
        self.test_precision = MulticlassPrecision(**metrics_kwargs)
        self.test_recall = MulticlassRecall(**metrics_kwargs)
        self.test_f1 = MulticlassF1Score(**metrics_kwargs)

    def forward(self, x):
        bname = self.hparams.backbone.lower()

        if "wav2vec2" in bname:
            hidden = self.backbone(x).last_hidden_state
        elif bname == "vit-imagenet":
            hidden = self.backbone(pixel_values=x).last_hidden_state
        elif bname == "mae-ast":
            if x.dim() == 3:
                x = x.squeeze(-1)

            output_dict = self.backbone(x)
            hidden = output_dict["hidden_states"][-1]
        else:
            raise ValueError(f"Unsupported backbone in forward(): '{bname}'")

        if bname == 'vit-imagenet':
            emb = hidden[:, 0]
        else:
            emb = hidden.mean(dim=1)

        return self.classifier(emb)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log_batch_metrics(loss, preds, y, prefix='train')
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log_batch_metrics(loss, preds, y, prefix='val')

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
        self.log_batch_metrics(loss, preds, y, prefix='test')

    def configure_optimizers(self):
        backbone_params = list(self.backbone.parameters())
        head_params     = list(self.classifier.parameters())
        optimizer = torch.optim.AdamW(
            [
                {"params": backbone_params, "lr": self.hparams.backbone_lr},
                {"params": head_params,     "lr": self.hparams.head_lr},
            ],
            weight_decay=self.hparams.weight_decay,
        )

        total_steps = self.trainer.estimated_stepping_batches

        warmup_steps = int(0.05 * total_steps)

        def lr_lambda(current_step: int):
            if current_step < warmup_steps:
                # linear warm‑up
                return float(current_step) / float(max(1, warmup_steps))
            # cosine decay after warm‑up
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return 0.5 * (1.0 + np.cos(np.pi * progress))

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}


    def log_batch_metrics(self, loss, preds, targets, prefix):
        self.log(f'{prefix}_loss', loss, prog_bar=True, on_epoch=True)
        acc = (preds == targets).float().mean()
        self.log(f'{prefix}_acc', acc, prog_bar=True, on_epoch=True)
        precision = getattr(self, f'{prefix}_precision')(preds, targets)
        recall = getattr(self, f'{prefix}_recall')(preds, targets)
        f1 = getattr(self, f'{prefix}_f1')(preds, targets)
        self.log(f'{prefix}_precision', precision, on_epoch=True)
        self.log(f'{prefix}_recall', recall, on_epoch=True)
        self.log(f'{prefix}_f1', f1, on_epoch=True)

    def on_train_end(self):
        save_dir = getattr(self, 'save_dir', None)
        if save_dir:
            self.save_model(save_dir)

    def save_model(self):
        base_dir = os.path.join('notebook', 'downstream')
        base_dir = os.path.join(base_dir, 'model')
        bn = self.hparams.backbone.replace('/', '_')
        cw = getattr(self.hparams, 'class_weights', None)
        balance_flag = 'imbalance' if cw is not None else 'balance'
        timestamp = getattr(self, 'finish_time', datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        folder = os.path.join(base_dir, f"{bn}_{balance_flag}", timestamp)
        os.makedirs(folder, exist_ok=True)

        ckpt_path = os.path.join(folder, f"{timestamp}.pt")
        payload = {
            'state_dict': self.state_dict(),
            'hparams': dict(self.hparams)
        }
        for attr in ('test_results', 'finish_time', 'epochs_trained'):
            if hasattr(self, attr):
                payload[attr] = getattr(self, attr)
        #torch.save(payload, ckpt_path)

        stats_path = os.path.join(folder, f"{timestamp}.txt")
        raw_hparams = dict(self.hparams)
        serializable_hparams = {}
        for k, v in raw_hparams.items():
            if isinstance(v, np.ndarray):
                serializable_hparams[k] = v.tolist()
            elif isinstance(v, torch.Tensor):
                serializable_hparams[k] = v.cpu().item() if v.ndim == 0 else v.cpu().tolist()
            else:
                serializable_hparams[k] = v

        serializable_results = {}
        if hasattr(self, 'test_results'):
            for k, v in self.test_results.items():
                serializable_results[k] = v.cpu().item() if isinstance(v, torch.Tensor) else v

        with open(stats_path, 'w') as f:
            f.write(f"Model architecture:\n{self}\n\n")
            f.write("Hyperparameters:\n")
            f.write(json.dumps(serializable_hparams, indent=4))
            f.write("\n\n")
            if serializable_results:
                f.write("Test results:\n")
                f.write(json.dumps(serializable_results, indent=4))
                f.write("\n\n")
            if hasattr(self, 'epochs_trained'):
                f.write(f"Epochs trained: {self.epochs_trained}\n")

        self._last_save_dir = folder
        self._last_timestamp = timestamp
        print(f"Artifacts saved to {folder}/")

    def load_mae_ckpt(self, ckpt_source: str):

        loaded = torch.load(ckpt_source)
        state_dict = loaded.get('model', loaded)

        up = self.backbone.upstream.model
        up_state = up.state_dict()

        to_load = {k: v for k, v in state_dict.items()
                   if k in up_state and v.shape == up_state[k].shape}

        missing, unexpected = up.load_state_dict(to_load, strict=False)

        for k, v in to_load.items():
            if not torch.equal(up_state[k], v):
                raise RuntimeError(f"Weight mismatch at '{k}' after loading checkpoint")

        print(f"Successfully loaded {len(to_load)} parameters; missing: {len(missing)}, unexpected: {len(unexpected)}")

    @classmethod
    def load_model(cls, load_dir: str, map_location=None):

        hparams_path = os.path.join(load_dir, 'hparams.json')
        with open(hparams_path, 'r') as f:
            hparams = json.load(f)

        model = cls(**hparams)
        ckpt_path = os.path.join(load_dir, f'{cls.__name__}.ckpt')
        state = torch.load(ckpt_path, map_location=map_location)
        model.load_state_dict(state['state_dict'])
        return model

def WMMD_Collate(batch):
    inputs, labels = zip(*batch)
    batch_inputs = torch.stack(inputs, dim=0)
    batch_labels = torch.tensor(labels, dtype=torch.long)
    return batch_inputs, batch_labels
class DBWithDeltas(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, spec):
        spec_db = F.amplitude_to_DB(
            spec,
            multiplier=10.0,
            amin=1e-10,
            db_multiplier=0
        )
        t = spec_db.transpose(1, 2)
        d1 = F.compute_deltas(t)
        d2 = F.compute_deltas(d1)
        return torch.cat([
            t.transpose(1, 2),
            d1.transpose(1, 2),
            d2.transpose(1, 2)
        ], dim=1)

class WMMDSoundDataset(torch.utils.data.Dataset):
    def __init__(self, dataset,
                 backbone: str,
                 target_sr: int = 16000,
                 max_audio_len_samples: int = 40000
                 ):
        """
        dataset: list of dicts with keys 'path' & 'label'
        backbone: 'facebook/wav2vec2-base' or 'mae-ast'
        target_sr: sampling rate (e.g. 2000)
        """

        self.target_size = 224
        self.n_fft = 4096
        self.win_length = 4096
        self.hop_length = 160
        self.n_mels = 128
        self.dataset = dataset
        self.backbone = backbone.lower()
        self.target_sr = target_sr
        self.max_audio_len_samples = max_audio_len_samples
        self.resampler_cache = {}

        if self.backbone in ["vit-imagenet",]:
            self.mel_transform = T.MelSpectrogram(
                sample_rate=self.target_sr, n_fft=self.n_fft, win_length=self.win_length, hop_length=self.hop_length,
                n_mels=self.n_mels
            )
            if self.backbone == "mae-ast":
                self.db_norm = DBWithDeltas()

        # Initialize model-specific processors
        if self.backbone == "vit-imagenet":
            self.processor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
        elif "wav2vec2" in self.backbone:
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.backbone, return_attention_mask=False,
                                                                      sampling_rate=self.target_sr)
        elif self.backbone == "mae-ast":
            self.processor = None
        else:
            raise ValueError(f"Unsupported backbone '{backbone}'")

        labels = sorted({item['label'] for item in dataset})
        self.label_to_int = {lbl: i for i, lbl in enumerate(labels)}

    @staticmethod
    def _safe_logmelspec(waveform, mel_spec, to_db, db_range=80.0):
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(1)
        spec = mel_spec(waveform) + 1e-10
        spec_db = to_db(spec)
        spec_db = torch.nan_to_num(spec_db, neginf=-db_range)
        t = spec_db
        d1 = F.compute_deltas(t.squeeze(1)).unsqueeze(1)
        d2 = F.compute_deltas(d1.squeeze(1)).unsqueeze(1)
        spec3 = torch.cat([t, d1, d2], dim=1)
        spec3 = ((spec3 + db_range)/db_range).clamp(0.0, 1.0)
        return spec3

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_path = item["path"]
        waveform, orig_sr = torchaudio.load(audio_path)

        if orig_sr != self.target_sr:
            if orig_sr not in self.resampler_cache:
                self.resampler_cache[orig_sr] = torchaudio.transforms.Resample(orig_sr, self.target_sr)
            waveform = self.resampler_cache[orig_sr](waveform)

        waveform = waveform / (waveform.abs().max() + 1e-6)
        wav_1d = waveform.squeeze(0)

        current_len = wav_1d.shape[0]
        if current_len > self.max_audio_len_samples:
            start = random.randint(0, current_len - self.max_audio_len_samples)
            wav_1d = wav_1d[start : start + self.max_audio_len_samples]
        elif current_len < self.max_audio_len_samples:
            pad_amt = self.max_audio_len_samples - current_len
            wav_1d = torch.nn.functional.pad(wav_1d, (0, pad_amt))

        required_len = self.n_fft
        if wav_1d.shape[0] < required_len:
            pad_amt = required_len - wav_1d.shape[0]
            wav_1d = torch.nn.functional.pad(wav_1d, (0, pad_amt))

        if "wav2vec2" in self.backbone:
            arr = wav_1d.numpy()
            feats = self.processor(arr, sampling_rate=self.target_sr, return_tensors="pt")
            inp = feats.input_values.squeeze(0)

        elif self.backbone == "mae-ast":
            inp = wav_1d

        elif self.backbone == "vit-imagenet":
            spec = self.mel_transform(wav_1d)
            spec_img = (spec - spec.min()) / (spec.max() - spec.min()) * 255
            spec_img = spec_img.to(torch.uint8).numpy()

            image_3_channel = np.stack([spec_img, spec_img, spec_img], axis=-1)

            inputs = self.processor(images=image_3_channel, return_tensors="pt")
            inp = inputs.pixel_values.squeeze(0)

        else:
            raise ValueError(f"Unsupported backbone '{self.backbone}'")

        lbl = self.label_to_int[item['label']]
        return inp, lbl

class WMMDDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dict,
        backbone: str,
        batch_size: int = 2,
        num_workers: int = 1,
    ):
        super().__init__()
        self.dataset_dict = dataset_dict
        self.backbone = backbone
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_audio_len_samples = 40000

    def setup(self, stage=None):
        ds_kwargs = {
            "backbone": self.backbone,
            'max_audio_len_samples': self.max_audio_len_samples,
        }
        self.train_ds = WMMDSoundDataset(self.dataset_dict["train"], **ds_kwargs)
        self.val_ds = WMMDSoundDataset(self.dataset_dict["validation"], **ds_kwargs)
        self.test_ds = WMMDSoundDataset(self.dataset_dict["test"], **ds_kwargs)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=WMMD_Collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=WMMD_Collate
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=WMMD_Collate
        )
#%%
# callback for logging metrics
class MetricsLogger(pl.Callback):
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        self.train_precisions = []
        self.val_precisions = []
        self.train_recalls = []
        self.val_recalls = []
        self.train_f1s = []
        self.val_f1s = []

    def on_train_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.train_losses.append(m['train_loss'].item())
        self.train_accs.append(m['train_acc'].item())
        self.train_precisions.append(m['train_precision'].item())
        self.train_recalls.append(m['train_recall'].item())
        self.train_f1s.append(m['train_f1'].item())

    def on_validation_epoch_end(self, trainer, pl_module):
        m = trainer.callback_metrics
        self.val_losses.append(m['val_loss'].item())
        self.val_accs.append(m['val_acc'].item())
        self.val_precisions.append(m['val_precision'].item())
        self.val_recalls.append(m['val_recall'].item())
        self.val_f1s.append(m['val_f1'].item())

# callback for early stopping
early_stopping = EarlyStopping(
    monitor='val_loss',
    mode='min',
    patience=10,
    min_delta=0.01,
    verbose=True
)
#%%
def train_finetune(cfg, early_stopping=False):
    # training Loops
    for cfg in cfg:
        dm = WMMDDataModule(
            dataset_dict=ds,
            backbone=cfg["backbone"],
            batch_size=2,
            num_workers=0,
        )

        model = WMMDClassifier(
            num_classes=cfg['num_classes'],
            backbone_lr=cfg['backbone_lr'],
            head_lr=cfg['head_lr'],
            weight_decay=cfg['weight_decay'],
            backbone=cfg['backbone'], finetune=cfg['finetune'],
            class_weights=cfg['class_weights'],
            ckpt_path=cfg['ckpt_path'],
            max_epochs=cfg['max_epochs']
        )
        metrics_cb = MetricsLogger()

        if early_stopping:
            callbacks = [metrics_cb, early_stopping]
        else:
            callbacks = [metrics_cb]

        trainer = pl.Trainer(
            max_epochs=cfg['max_epochs'],
            accelerator='gpu', devices=1,
            precision='16-mixed', accumulate_grad_batches=2,
            check_val_every_n_epoch=1,
            num_sanity_val_steps=0,
            enable_progress_bar=False, log_every_n_steps=50,
            enable_checkpointing=False,
            callbacks=callbacks
        )

        trainer.fit(model, dm)
        test_res = trainer.test(model, dm)[0]

        model.test_results = test_res
        model.finish_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model.epochs_trained = trainer.current_epoch + 1

        model.save_model()

        metrics_map = {
            'accuracy':   ('train_accs',      'val_accs'),
            'precision':  ('train_precisions','val_precisions'),
            'recall':     ('train_recalls',   'val_recalls'),
            'f1_score':   ('train_f1s',       'val_f1s'),
        }

        for metric_name, (train_attr, val_attr) in metrics_map.items():
            train_vals = getattr(metrics_cb, train_attr)
            val_vals = getattr(metrics_cb, val_attr)
            epochs = list(range(1, len(train_vals) + 1))

            plt.figure()
            plt.plot(epochs, train_vals, label=f'train_{metric_name}')
            plt.plot(epochs, val_vals,   label=f'val_{metric_name}')
            plt.xlabel('Epoch')
            plt.ylabel(metric_name.replace('_', ' ').title())
            plt.title(f"{metric_name.replace('_', ' ').title()} over Epochs {model._last_timestamp}")
            plt.grid(True)
            plt.legend(loc='best')

            plot_file = os.path.join(model._last_save_dir, f"{model._last_timestamp}_{metric_name}.png")
            plt.savefig(plot_file)
            plt.close()

        print(f"Completed {cfg['backbone']} ({'FT' if cfg['finetune'] else 'Frozen'}), artifacts in {model._last_save_dir}")


def aggregate_metrics(model_name, dir, start_time):
    print(f"\nAggregating results from '{dir}'...")

    csv_files_from_this_run = []
    for root, dirs, files in os.walk(dir):
        if "metrics.csv" in files:
            # Check if the parent 'version_X' folder was created after our run started
            dir_creation_time = datetime.datetime.fromtimestamp(os.path.getctime(root))
            if dir_creation_time > start_time:
                csv_files_from_this_run.append(os.path.join(root, "metrics.csv"))

    if not csv_files_from_this_run:
        print("No recent 'metrics.csv' files found from this run.")
    else:
        print(f"Found {len(csv_files_from_this_run)} relevant log files to aggregate.")

        all_metrics_dfs = [pd.read_csv(f) for f in csv_files_from_this_run]
        stacked_data = np.stack([df.to_numpy() for df in all_metrics_dfs], axis=0)

        mean_data = np.nanmean(stacked_data, axis=0)
        std_data = np.nanstd(stacked_data, axis=0)

        mean_df = pd.DataFrame(mean_data, columns=all_metrics_dfs[0].columns)
        std_df = pd.DataFrame(std_data, columns=all_metrics_dfs[0].columns)

        mean_df.to_csv(f"notebook/downstream/ft_lp_results/_aggregate results/{model_name}-mean.csv", index=False)
        std_df.to_csv(f"notebook/downstream/ft_lp_results/_aggregate results/{model_name}-std.csv", index=False)
        print(f"\nAggregated results saved to 'notebook/downstream/ft_lp_results/_aggregate results/{model_name}-mean.csv' and 'notebook/downstream/ft_lp_results/_aggregate results/{model_name}-std.csv'")