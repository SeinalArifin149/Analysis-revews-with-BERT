from slank import slang_dict
import re

import pandas as pd
df = pd.read_csv("Bukit jaddih.csv")

df = df.drop(columns=["title","url","name","reviewUrl"])

df = df.dropna(subset=["text"])  # wajib ada text
df = df[df["text"].str.strip() != ""]  # hapus yang kosong tapi bukan NaN
df = df.reset_index(drop=True)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)      # hapus URL
    text = re.sub(r"@\w+", "", text)         # hapus mention
    text = re.sub(r"#\w+", "", text)         # hapus hashtag
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # hapus angka & simbol
    text = re.sub(r"\s+", " ", text).strip() # hapus spasi berlebih
    return text

# 2. CASE FOLDING
def lower(text):
    return text.lower()

# 3. SLANG NORMALIZATION
def slank_normalization(text):
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

# 4. PIPELINE (biar enak dipanggil)
def preprocess(text):
    text = str(text)    # pastikan string
    text = clean_text(text)
    text = lower(text)
    text = slank_normalization(text)
    return text