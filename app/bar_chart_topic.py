import streamlit as st
import matplotlib.pyplot as plt

def topik_positif_bar_chart(filtered_df):

    positif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "positive"
    ]

    # Diurutkan dari yang terkecil agar batang terpanjang ada di atas
    topik_count = positif_df["nama_topik"].value_counts().sort_values(ascending=True)

    if topik_count.empty:
        st.info("Tidak ada data topik positif.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Membuat Horizontal Bar Chart (Warna Hijau)
    bars = ax.barh(topik_count.index, topik_count.values, color='#2ca02c')

    ax.set_title("Topik Sentimen Positif", fontsize=12, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=9) 
    ax.tick_params(axis='x', labelsize=8)

    # TRICK PRO: Melebarkan batas kanan sumbu X agar teks "ulasan" tidak kepotong
    max_value = topik_count.values.max()
    ax.set_xlim(0, max_value + (max_value * 0.25))

    # Menambahkan label angka
    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max_value * 0.02), # Jaraknya disesuaikan biar rapi
            bar.get_y() + bar.get_height() / 2, 
            f'{int(width)} ulasan', 
            va='center', ha='left', fontsize=8, color='black'
        )

    # Menghilangkan garis tepi kanan dan atas
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # Mencegah memory leak di Streamlit


def topik_negatif_bar_chart(filtered_df):

    negatif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "negative"
    ]

    topik_count = negatif_df["nama_topik"].value_counts().sort_values(ascending=True)

    if topik_count.empty:
        st.info("Tidak ada data topik negatif.")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    # Membuat Horizontal Bar Chart (Warna Merah)
    bars = ax.barh(topik_count.index, topik_count.values, color='#d62728')

    ax.set_title("Topik Sentimen Negatif", fontsize=12, fontweight='bold', pad=15)
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)

    # TRICK PRO: Melebarkan batas kanan sumbu X
    max_value = topik_count.values.max()
    ax.set_xlim(0, max_value + (max_value * 0.25))

    for bar in bars:
        width = bar.get_width()
        ax.text(
            width + (max_value * 0.02), 
            bar.get_y() + bar.get_height() / 2, 
            f'{int(width)} ulasan', 
            va='center', ha='left', fontsize=8, color='black'
        )

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig) # Mencegah memory leak