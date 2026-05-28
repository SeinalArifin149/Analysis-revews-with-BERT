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

# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Dashboard Sentimen",
    page_icon="📊",
    layout="wide"
)

# ======================
# LOAD DATA
# ======================
df = load_data()

# ======================
# HEADER
# ======================
header()

# ======================
# LABEL TOPIK
# ======================
topic_label = {
    -1: "Umum / Campuran",
     0: "Keindahan Wisata Religi",
     1: "Akses dan tiket wisata",
     2: "Makam Syaikhona Kholil",
     3: "Doa dan religius"
}

df["nama_topik"] = df["hasil_topik"].map(topic_label)

# ======================
# FILTER
# ======================
sentiment_option, topik_option, star_range = filter_sidebar(df)

filtered_df = df[
    (df["hasil_sentimen"].isin(sentiment_option)) &
    (df["nama_topik"].isin(topik_option)) &
    (df["stars"] >= star_range[0]) &
    (df["stars"] <= star_range[1])
]

# ======================
# METRIC
# ======================
show_metric(filtered_df)

# ======================
# SENTIMEN (CENTER)
# ======================
col_left, col_center, col_right = st.columns([1, 2, 1])

with col_center:
    st.markdown("### 📊 Distribusi Sentimen")
    sentimen_pie_chart(filtered_df)

# ======================
# TOPIK (KANAN & KIRI BALANCE)
# ======================
positive_exist = (filtered_df["hasil_sentimen"] == "positive").any()
negative_exist = (filtered_df["hasil_sentimen"] == "negative").any()

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

# ======================
# TABLE
# ======================
st.markdown("## 📄 Tabel Data")

st.dataframe(filtered_df, use_container_width=True)