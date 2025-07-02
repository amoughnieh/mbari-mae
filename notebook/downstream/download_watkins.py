import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
import urllib.request
from zipfile import ZipFile

DATA_URL    = "https://storage.googleapis.com/ml-bioacoustics-datasets/watkins.zip"
ZIP_PATH    = Path("data") / "watkins.zip"
EXTRACT_DIR = Path("data") / "watkins"
EXTRACT_DIR.parent.mkdir(parents=True, exist_ok=True)

if not ZIP_PATH.exists():
    print(f"Downloading to {ZIP_PATH}…")
    urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

if not (EXTRACT_DIR / "annotations.csv").exists():
    print(f"Extracting to {EXTRACT_DIR}…")
    with ZipFile(ZIP_PATH, "r") as zp:
        zp.extractall(EXTRACT_DIR)

df = pd.read_csv('data/watkins/annotations.csv')
df = df.apply(lambda r: pd.Series({'path': f"data/watkins/{r['path']}", 'label': r['species']}), axis=1)

df = df[df['label'] != 'Weddell_Seal']    # remove Weddell Seal which has only two instances
# split to train:valid:test = 6:2:2
df_train, df_valid_test = train_test_split(df, test_size=0.4, random_state=42, shuffle=True, stratify=df['label'])
df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, random_state=42, shuffle=True, stratify=df_valid_test['label'])
df_train_low, _ = train_test_split(df_train, test_size=0.8, random_state=42, shuffle=True, stratify=df_train['label'])

df_train = df_train.sort_index()
df_train_low = df_train_low.sort_index()
df_valid = df_valid.sort_index()
df_test = df_test.sort_index()

df_train.to_csv('data/watkins/annotations.train.csv')
df_train_low.to_csv('data/watkins/annotations.train-low.csv')
df_valid.to_csv('data/watkins/annotations.valid.csv')
df_test.to_csv('data/watkins/annotations.test.csv')
