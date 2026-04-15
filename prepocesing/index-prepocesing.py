from pathlib import Path
from func_prepocesing import preprocess, load_and_clean
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

print("skrip jalan bg")
df = load_and_clean(BASE_DIR / "Bukit jaddih.csv")

# Lewati baris yang text-nya null/NaN
df = df.dropna(subset=["text"])

# Rapikan, lalu buang yang kosong
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""]

df["clean_text"] = df["text"].apply(preprocess)
df.to_csv(BASE_DIR / "hasil_prepocesing.csv", index=False)

print("Done Preprocessingnya bang")

