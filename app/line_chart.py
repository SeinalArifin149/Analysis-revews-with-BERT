import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

def line_chart_sentimen_topik(filtered_df):
    # 1. Kelompokkan data berdasarkan topik dan sentimen
    df_grouped = filtered_df.groupby(['nama_topik', 'hasil_sentimen']).size().unstack(fill_value=0)
    
    # 2. Pastikan kolom positive & negative selalu ada (mencegah error)
    if 'positive' not in df_grouped.columns:
        df_grouped['positive'] = 0
    if 'negative' not in df_grouped.columns:
        df_grouped['negative'] = 0
        
    # 3. Hitung total ulasan per topik untuk mengurutkan sumbu X
    # Diurutkan dari ulasan terbanyak ke terdikit agar garisnya menurun rapi
    df_grouped['total'] = df_grouped['positive'] + df_grouped['negative']
    df_grouped = df_grouped.sort_values('total', ascending=False)
    
    if df_grouped.empty:
        st.info("Tidak ada data untuk ditampilkan.")
        return

    # 4. Mulai Menggambar Line Chart
    fig, ax = plt.subplots(figsize=(12, 6)) # Dibikin agak panjang biar teks X nggak dempet
    
    topics = df_grouped.index
    pos_counts = df_grouped['positive']
    neg_counts = df_grouped['negative']
    
    # Garis Positif (Hijau)
    ax.plot(topics, pos_counts, color='#2ca02c', marker='o', label='Positif', linewidth=2.5)
    # Garis Negatif (Merah)
    ax.plot(topics, neg_counts, color='#d62728', marker='o', label='Negatif', linewidth=2.5)
    
    ax.set_title("Komparasi Tren Sentimen per Topik", fontsize=15, fontweight='bold', pad=20)
    ax.set_xlabel("Topik Wisata", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Jumlah Ulasan", fontsize=11, fontweight='bold')
    
    # Miringkan tulisan topik di sumbu X biar nggak tabrakan
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(fontsize=10)
    
    # Percantik tampilan: hapus garis tepi, tambah garis bantu
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Tambahkan angka kecil di tiap titik/bundaran
    for i, txt in enumerate(pos_counts):
        if txt > 0:
            ax.text(i, txt + (max(pos_counts)*0.02), str(txt), ha='center', va='bottom', fontsize=8, color='#2ca02c', fontweight='bold')
            
    for i, txt in enumerate(neg_counts):
        if txt > 0:
            ax.text(i, txt + (max(neg_counts)*0.02), str(txt), ha='center', va='bottom', fontsize=8, color='#d62728', fontweight='bold')
            
    ax.legend()
    plt.tight_layout()
    
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)