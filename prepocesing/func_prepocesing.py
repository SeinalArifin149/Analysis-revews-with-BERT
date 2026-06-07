from slank import slang_dict
import re
import pandas as pd
import emoji
from langdetect import detect
from deep_translator import GoogleTranslator
import nltk 
from nltk.corpus import stopwords

# matiin bang kalau nak coba tanpa stop word
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

def translate(text):
    try:
        lang = detect(text)
        if lang == "id":
            return text
        
        translated = GoogleTranslator(source="auto", target="id").translate(text)
        return translated
    except:
        return text

def load_and_clean(path):
    print("bersih bersih dulu gan...")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=["url", "name", "reviewurl"], errors="ignore")
    df = df.dropna(subset=["text", "stars"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)
    return df

def remove_emoji(text):
    return emoji.replace_emoji(text, replace='')

def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#\w+", "", text)
    text = remove_emoji(text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def lower(text):
    return text.lower()

def stopword(text):
    stop_words = set(stopwords.words('indonesian'))
    
    # Custom stopword (sudah termasuk "melalui")
    tambahan_stopword = {"melalui"} 
    stop_words.update(tambahan_stopword)
    
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    hasil = " ".join(filtered_words)
    return hasil

def slank_normalization(text):
    words = text.split()
    normalized_words = [slang_dict.get(word, word) for word in words]
    return " ".join(normalized_words)

def preprocess(text):
    text = str(text)
    
    print("\n" + "="*50)
    print(f"[0] Original       : '{text}'")
    
    text = translate(text)
    print(f"[1] After Translate: '{text}'")
    
    text = clean_text(text)
    print(f"[2] After Clean    : '{text}'")
    
    text = lower(text)
    print(f"[3] After Lowercase: '{text}'")
    
    text = stopword(text)
    print(f"[4] After Stopword : '{text}'")
    
    text = slank_normalization(text)
    print(f"[5] After Slang    : '{text}'")
    print("="*50 + "\n")
    
    return text