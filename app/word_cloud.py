import streamlit as st
import matplotlib.pyplot as plt

from wordcloud import WordCloud


def wordcloud_positive(filtered_df):

    # ======================
    # FILTER POSITIVE
    # ======================
    positive_df = filtered_df[
        filtered_df["hasil_sentimen"] == "positive"
    ]

    # ======================
    # CEK DATA KOSONG
    # ======================
    if positive_df.empty:
        return

    # ======================
    # GABUNGKAN SEMUA TEKS
    # ======================
    text = " ".join(
        positive_df["teks"].astype(str)
    )

    # ======================
    # TOKENISASI
    # ======================
    tokens = text.split()

    # gabung lagi
    final_text = " ".join(tokens)

    # ======================
    # WORDCLOUD
    # ======================
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(final_text)

    # ======================
    # PLOT
    # ======================
    fig, ax = plt.subplots(figsize=(8,4))

    ax.imshow(wordcloud, interpolation="bilinear")

    ax.axis("off")

    ax.set_title(
        "Wordcloud Sentimen Positif",
        fontsize=14
    )

    st.pyplot(fig)


def wordcloud_negative(filtered_df):

    # ======================
    # FILTER NEGATIVE
    # ======================
    negative_df = filtered_df[
        filtered_df["hasil_sentimen"] == "negative"
    ]

    # ======================
    # CEK DATA KOSONG
    # ======================
    if negative_df.empty:
        return

    # ======================
    # GABUNGKAN TEKS
    # ======================
    text = " ".join(
        negative_df["teks"].astype(str)
    )

    # ======================
    # TOKENISASI
    # ======================
    tokens = text.split()

    final_text = " ".join(tokens)

    # ======================
    # WORDCLOUD
    # ======================
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white"
    ).generate(final_text)

    # ======================
    # PLOT
    # ======================
    fig, ax = plt.subplots(figsize=(8,4))

    ax.imshow(wordcloud, interpolation="bilinear")

    ax.axis("off")

    ax.set_title(
        "Wordcloud Sentimen Negatif",
        fontsize=14
    )

    st.pyplot(fig)