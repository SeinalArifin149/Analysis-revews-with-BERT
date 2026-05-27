import streamlit as st
from load_data import load_data

def filter_sidebar():

    # ======================
    # LOAD DATA
    # ======================
    df = load_data()

    # ======================
    # LABEL TOPIK
    # ======================
    topic_labels = {
        -1: "📌 Umum / Campuran",
         0: "🏝️ Keindahan Wisata Religi",
         1: "🚗 Akses dan Tiket Wisata",
         2: "🕌 Makam Syaikhona Kholil",
         3: "🤲 Doa dan Religius"
    }

    # tambah kolom nama topik
    df["nama_topik"] = df["hasil_topik"].map(topic_labels)

    # ======================
    # SIDEBAR TITLE
    # ======================
    st.sidebar.title("Filter Data")

    # ======================
    # FILTER SENTIMEN
    # ======================
    sentiment_option = st.sidebar.multiselect(
        "Pilih Sentimen",
        options=df["hasil_sentimen"].unique(),
        default=df["hasil_sentimen"].unique()
    )

    # ======================
    # FILTER TOPIK
    # ======================
    topik_option = st.sidebar.multiselect(
        "Pilih Topik",
        options=df["nama_topik"].unique(),
        default=df["nama_topik"].unique()
    )

    # ======================
    # FILTER RATING
    # ======================
    min_star = int(df["stars"].min())
    max_star = int(df["stars"].max())

    star_range = st.sidebar.slider(
        "Filter Rating",
        min_value=min_star,
        max_value=max_star,
        value=(min_star, max_star)
    )

    # ======================
    # FILTER DATAFRAME
    # ======================
    filtered_df = df[
        (df["hasil_sentimen"].isin(sentiment_option)) &
        (df["nama_topik"].isin(topik_option)) &
        (df["stars"] >= star_range[0]) &
        (df["stars"] <= star_range[1])
    ]

    return filtered_df