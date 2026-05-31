import streamlit as st
import matplotlib.pyplot as plt

def sentimen_pie_chart(filtered_df):

    sentimen_count = filtered_df["hasil_sentimen"].value_counts()

    if sentimen_count.empty:
        st.info("Tidak ada data sentimen.")
        return

    # 1. Ukuran dikecilkan ke mode proporsional (sweet spot)
    fig, ax = plt.subplots(figsize=(5, 5))

    colors = ['#2ca02c' if label == 'positive' else '#d62728' for label in sentimen_count.index]

    ax.pie(
        sentimen_count,
        labels=sentimen_count.index,
        autopct='%1.1f%%',
        radius=0.9,  # Radius diturunkan sedikit biar nggak mentok tepi canvas
        colors=colors,
        textprops={'fontsize': 10},  # Font dikecilkan lagi
        startangle=90, 
        explode=[0.05] * len(sentimen_count) 
    )

    ax.set_title(
        "Distribusi Sentimen Keseluruhan",
        fontsize=12, # Font judul disesuaikan
        fontweight='bold',
        pad=15
    )

    plt.tight_layout()

    # 2. Hapus use_container_width=True agar grafiknya mempertahankan ukuran aslinya
    st.pyplot(fig)
    plt.close(fig)