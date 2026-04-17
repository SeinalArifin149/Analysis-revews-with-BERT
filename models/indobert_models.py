import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==============================
# 1. LOAD DATA
# ==============================
file_path = "../Bukit Jaddih(Cleaning).csv"
df = pd.read_csv(file_path)

texts = df["teks"].astype(str).tolist()

# ==============================
# 2. LOAD MODEL SENTIMENT
# ==============================
model_name = "w11wo/indonesian-roberta-base-sentiment-classifier"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

model.eval()

# label mapping
label_map = {
    0: "negative",
    1: "neutral",
    2: "positive"
}

print("Model siap bang 🔥")

# ==============================
# 3. FUNCTION PREDICT
# ==============================
def predict_sentiment(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    pred_class = torch.argmax(logits, dim=1).item()

    return label_map[pred_class]

# ==============================
# 4. PROSES SEMUA DATA
# ==============================
results = []

print("Mulai proses bang...")

for i, text in enumerate(texts):
    sentiment = predict_sentiment(text)
    results.append(sentiment)

    # tampilkan 5 data pertama
    if i < 5:
        print("\n======")
        print(f"Text     : {text}")
        print(f"Sentimen : {sentiment}")

# ==============================
# 5. SIMPAN HASIL
# ==============================
df["sentiment"] = results

output_path = "../hasil_sentiment_indobert.csv"
df.to_csv(output_path, index=False)

print("\nSelesai bang 🔥🔥🔥")