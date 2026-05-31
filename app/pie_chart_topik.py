import streamlit as st
import matplotlib.pyplot as plt

def topik_positif_chart(filtered_df):

    positif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "positive"
    ]

    topik_count = positif_df["nama_topik"].value_counts()

    if topik_count.empty:
        return

    # 1. Lebarkan ukuran grafiknya jadi 7x4 agar kotak Legend muat
    fig, ax = plt.subplots(figsize=(7, 4))

    # 2. Buat pie chart tanpa label teks di luarnya
    wedges, texts, autotexts = ax.pie(
        topik_count,
        # Hanya tampilkan persentase jika ukurannya lebih dari 3.5%
        autopct=lambda p: f'{p:.1f}%' if p > 3.5 else '', 
        radius=1,
        startangle=140,
        textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1} # Tambah garis putih pemisah
    )

    # 3. Pindahkan daftar nama topik ke Legend di sebelah kanan
    ax.legend(
        wedges, 
        topik_count.index,
        title="Daftar Topik",
        loc="center left",
        bbox_to_anchor=(1, 0.5), # Menggeser kotak Legend ke kanan luar lingkaran
        fontsize=8,
        title_fontsize=9
    )

    ax.set_title("Topik Sentimen Positif", fontsize=12, pad=10)

    # Mencegah legend terpotong oleh layout
    plt.tight_layout()

    st.pyplot(fig)


def topik_negatif_chart(filtered_df):

    negatif_df = filtered_df[
        filtered_df["hasil_sentimen"] == "negative"
    ]

    topik_count = negatif_df["nama_topik"].value_counts()

    if topik_count.empty:
        return

    fig, ax = plt.subplots(figsize=(7, 4))

    wedges, texts, autotexts = ax.pie(
        topik_count,
        autopct=lambda p: f'{p:.1f}%' if p > 3.5 else '',
        radius=1,
        startangle=140,
        textprops={'fontsize': 8, 'color': 'white', 'weight': 'bold'},
        wedgeprops={'edgecolor': 'white', 'linewidth': 1}
    )

    ax.legend(
        wedges, 
        topik_count.index,
        title="Daftar Topik",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=8,
        title_fontsize=9
    )

    ax.set_title("Topik Sentimen Negatif", fontsize=12, pad=10)

    plt.tight_layout()

    st.pyplot(fig)