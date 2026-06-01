import streamlit as st

def filter_sidebar(df):
    
    # ========================================================
    # 1. NAVIGASI CEPAT (Pindah ke atas)
    # ========================================================
    st.sidebar.markdown("## 📍 Chart")
    
    st.sidebar.markdown("""
    * [📊 Analisis Sentimen Keseluruhan (Pie Chart)](#distribusi-sentimen-keseluruhan)
    * [🥧 Analisis Topik Wisata (Pie Chart)](#analisis-topik-wisata-pie-chart)
    * [📊 Analisis Topik Wisata (Bar Chart)](#analisis-topik-wisata-bar-chart)
    * [📈 Komparasi Sentimen per Topik (Line Chart)](#komparasi-sentimen-per-topik)
    * [☁️ Wordcloud Sentimen](#wordcloud-sentimen)
    """)

    # Garis pembatas biar rapi
    st.sidebar.markdown("---") 

    # ========================================================
    # 2. FILTER DATA (Pindah ke bawah)
    # ========================================================
    st.sidebar.markdown("## ⚙️ Filter Data")
    
    # Sentiment
    sentiment_option = st.sidebar.multiselect(
        "Pilih Sentimen",
        options=df["hasil_sentimen"].unique(),
        default=df["hasil_sentimen"].unique()
    )
    
    # Topik
    topik_option = st.sidebar.multiselect(
        "Pilih Topik",
        options=df["nama_topik"].unique(),
        default=df["nama_topik"].unique()
    )
    
    # Star
    min_star = int(df["stars"].min())
    max_star = int(df["stars"].max())

    star_range = st.sidebar.slider(
        "Filter Rating",
        min_value=min_star,
        max_value=max_star,
        value=(min_star, max_star)
    )

    return sentiment_option, topik_option, star_range