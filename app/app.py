import streamlit as st

from load_data import load_data
from sidebar import filter_sidebar
from header import header

from metric import show_metric

from pie_chart import sentimen_pie_chart
from pie_chart_topik import (
    topik_positif_chart,
    topik_negatif_chart
)
from bar_chart_topic import (
    topik_positif_bar_chart,
    topik_negatif_bar_chart
)
from word_cloud import wordcloud_negative, wordcloud_positive

st.set_page_config(
    page_title="Dashboard Sentimen",
    page_icon="📊",
    layout="wide"
)

df = load_data()

header()

# ========================================================
# 👇 MAPPING TOPIK BERDASARKAN HASIL BERTOPIC ASLI 👇
# ========================================================
topic_label = {
    -1: "Lainnya / Noise",                         
     0: "Keindahan & Spot Foto",                   
     1: "Wisata Religi (Makam Syaikhona Kholil)",  
     2: "Wisata Alam (Bukit Kapur Jaddih)",        
     3: "Wisata Religi & Kegiatan Umum",           
     4: "Lainnya / Noise",                         
     5: "Keluhan Pungli & Pengemis",               
     6: "Infrastruktur & Akses Jalan",             
     7: "Lanskap Tambang Kapur",                   
     8: "Lainnya / Noise",                         
     9: "Kondisi Alam & Danau",                    
    10: "Lainnya / Noise",                         
    11: "Kebersihan & Area Asri",                  
    12: "Keindahan Pemandangan",                   
    13: "Kepadatan Peziarah",                      
    14: "Lainnya / Noise",                         
    15: "Fasilitas (Parkir & Toilet)",             
    16: "Suasana Tenang & Damai",                  
    17: "Penataan Lokasi"                          
}

df["nama_topik"] = df["hasil_topik"].map(topic_label)

sentiment_option, topik_option, star_range = filter_sidebar(df)

filtered_df = df[
    (df["hasil_sentimen"].isin(sentiment_option)) &
    (df["nama_topik"].isin(topik_option)) &
    (df["stars"] >= star_range[0]) &
    (df["stars"] <= star_range[1])
]

show_metric(filtered_df)

st.markdown("---")

# ========================================================
# 1. PIE CHART SENTIMEN KESELURUHAN
# ========================================================
st.markdown("<h3 style='text-align: center;'>📊 Distribusi Sentimen Keseluruhan</h3>", unsafe_allow_html=True)

col_kiri, col_tengah, col_kanan = st.columns([1, 2, 1])
with col_tengah:
    sentimen_pie_chart(filtered_df)

st.markdown("---")

# Cek ketersediaan sentimen untuk bagian topik & wordcloud
positive_exist = (filtered_df["hasil_sentimen"] == "positive").any()
negative_exist = (filtered_df["hasil_sentimen"] == "negative").any()

# ========================================================
# 2. ANALISIS TOPIK WISATA (BAR CHART)
# ========================================================

st.markdown("## 🥧 Analisis Topik Wisata (Pie Chart)")

if positive_exist and negative_exist:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 😊 Topik Positif")
        topik_positif_chart(filtered_df)
    with col2:
        st.markdown("### 😡 Topik Negatif")
        topik_negatif_chart(filtered_df)
elif positive_exist:
    st.markdown("### 😊 Topik Positif")
    topik_positif_chart(filtered_df)
elif negative_exist:
    st.markdown("### 😡 Topik Negatif")
    topik_negatif_chart(filtered_df)
    
st.markdown("## 📈 Analisis Topik Wisata (Bar Chart)")

if positive_exist and negative_exist:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 😊 Topik Positif")
        topik_positif_bar_chart(filtered_df)
    with col2:
        st.markdown("### 😡 Topik Negatif")
        topik_negatif_bar_chart(filtered_df)
elif positive_exist:
    st.markdown("### 😊 Topik Positif")
    topik_positif_bar_chart(filtered_df)
elif negative_exist:
    st.markdown("### 😡 Topik Negatif")
    topik_negatif_bar_chart(filtered_df)

st.markdown("---")

# ========================================================
# 3. ANALISIS TOPIK WISATA (PIE CHART)
# ========================================================

st.markdown("---")

# ========================================================
# 4. WORDCLOUD
# ========================================================
st.markdown("## ☁️ Wordcloud Sentimen")

if positive_exist and negative_exist:
    col1, col2 = st.columns(2)
    with col1:
        wordcloud_positive(filtered_df)
    with col2:
        wordcloud_negative(filtered_df)
elif positive_exist:
    wordcloud_positive(filtered_df)
elif negative_exist:
    wordcloud_negative(filtered_df)

# st.markdown("## 📄 Tabel Data")
# st.dataframe(filtered_df, use_container_width=True)