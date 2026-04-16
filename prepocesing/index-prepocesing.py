from pathlib import Path
from func_prepocesing import preprocess, load_and_clean

BASE_DIR = Path(__file__).resolve().parent

print("skrip jalan bg")
df = load_and_clean(BASE_DIR / "Bukit jaddih.csv")
df["clean_text"] = df["text"].apply(preprocess)
df = df.drop(columns=["text"]) 
df.to_csv(BASE_DIR / "Bukit Jaddih (After Cleaning", index=False)
print("Done Preprocessingnya bang")

