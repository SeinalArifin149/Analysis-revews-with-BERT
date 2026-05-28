import streamlit as st
import matplotlib.pyplot as plt


def topik_positif_chart(filtered_df):

    positif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "positive"
    ]

    topik_count = positif_df["nama_topik"].value_counts()

    if topik_count.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5,3.5))

    ax.pie(
        topik_count,
        labels=topik_count.index,
        autopct='%1.1f%%',
        radius=0.75,
        textprops={'fontsize':8}
    )

    ax.set_title("Topik Sentimen Positif", fontsize=10)

    plt.tight_layout()

    st.pyplot(fig)


def topik_negatif_chart(filtered_df):

    negatif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "negative"
    ]

    topik_count = negatif_df["nama_topik"].value_counts()

    if topik_count.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5,3.5))

    ax.pie(
        topik_count,
        labels=topik_count.index,
        autopct='%1.1f%%',
        radius=0.75,
        textprops={'fontsize':8}
    )

    ax.set_title("Topik Sentimen Negatif", fontsize=10)

    plt.tight_layout()

    st.pyplot(fig)