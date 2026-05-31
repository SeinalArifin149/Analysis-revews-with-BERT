import streamlit as st
import matplotlib.pyplot as plt
import re

from wordcloud import WordCloud


# ======================
# CLEAN TEXT
# ======================
def clean_text(text):

    stopwords = {
        "yang", "tidak", "saja", "nya", "sudah",
        "dan", "di", "ke", "dari", "untuk",
        "dengan", "ada", "ini", "itu", "juga",
        "karena", "pada", "dalam", "atau",
        "sangat", "lebih", "bisa", "masih",
        "jadi", "agar", "seperti", "cukup",
        "sangat", "kalau", "semua",

        # kata informal
        "yg", "ga", "gak", "nggak", "aja",
        "nih", "banget", "udah", "sih",
        "kok", "lah", "deh", "dong",

        # kata umum wisata
        "tempat", "wisata"
    }

    # lowercase
    text = text.lower()

    # hapus angka
    text = re.sub(r"\d+", " ", text)

    # hapus tanda baca
    text = re.sub(r"[^\w\s]", " ", text)

    # tokenisasi
    tokens = text.split()

    # hapus stopword dan kata pendek
    tokens = [
        token
        for token in tokens
        if token not in stopwords
        and len(token) > 1
    ]

    return " ".join(tokens)


# ======================
# WORDCLOUD POSITIVE
# ======================
def wordcloud_positive(filtered_df):

    positive_df = filtered_df[
        filtered_df["hasil_sentimen"] == "positive"
    ]

    if positive_df.empty:
        return

    text = " ".join(
        positive_df["teks"].astype(str)
    )

    final_text = clean_text(text)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(final_text)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    ax.set_title(
        "Wordcloud Sentimen Positif",
        fontsize=14
    )

    plt.tight_layout()

    st.pyplot(fig)


# ======================
# WORDCLOUD NEGATIVE
# ======================
def wordcloud_negative(filtered_df):

    negative_df = filtered_df[
        filtered_df["hasil_sentimen"] == "negative"
    ]

    if negative_df.empty:
        return

    text = " ".join(
        negative_df["teks"].astype(str)
    )

    final_text = clean_text(text)

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color="white"
    ).generate(final_text)

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.imshow(
        wordcloud,
        interpolation="bilinear"
    )

    ax.axis("off")

    ax.set_title(
        "Wordcloud Sentimen Negatif",
        fontsize=14
    )

    plt.tight_layout()

    st.pyplot(fig)