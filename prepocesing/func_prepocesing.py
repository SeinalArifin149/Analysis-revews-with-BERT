from slank import slang_dict
import re

import pandas as pd
def load_and_clean(path):
    df = pd.read_csv(path)
    
    df.columns = df.columns.str.strip()
    print("Kolom awal:", df.columns)

    df = df.drop(columns=["title","url","name","reviewUrl"], errors='ignore')

    print("Setelah drop:", df.columns)

    df = df.dropna(subset=["text","stars"])
    df = df[df["text"].str.strip() != ""]
    df = df.reset_index(drop=True)

    return df
# def remove_emoji(text):
#     emoji_pattern = re.compile(
#         "["
#         u"\U0001F600-\U0001F64F"  # emoticon
#         u"\U0001F300-\U0001F5FF"  # simbol & pictograph
#         u"\U0001F680-\U0001F6FF"  # transport
#         u"\U0001F1E0-\U0001F1FF"  # bendera
#         "]+",
#         flags=re.UNICODE
#     )
#     return emoji_pattern.sub("", text)

def clean_text(text):
    text = re.sub(r"http\S+", "", text)      # hapus URL
    text = re.sub(r"@\w+", "", text)         # hapus mention
    text = re.sub(r"#\w+", "", text)         # hapus hashtag
    # text = remove_emoji(text)   
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