import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# LOAD MODEL YANG SUDAH ADA
# ==============================
model_path = "./model_sentiment_binary"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_map = {
    0: "negative",
    1: "positive"
}

def predict(text):
    inputs = tokenizer(
        str(text),
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    pred = torch.argmax(outputs.logits, dim=1).item()
    return label_map[pred]

# ==============================
# LOAD DATA
# ==============================
df = pd.read_csv("../Bukit Jaddih(Cleaning).csv")

df = df.dropna(subset=["teks"])

# ==============================
# PREDIKSI
# ==============================
df["prediksi"] = df["teks"].apply(predict)

# ==============================
# SIMPAN CSV
# ==============================
df.to_csv("hasil_prediksi.csv", index=False)

print("🔥 CSV berhasil dibuat!")