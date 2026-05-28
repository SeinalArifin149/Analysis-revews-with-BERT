import streamlit as st

def filter_sidebar(df):
    # Sentiment
    sentiment_option = st.sidebar.multiselect(
        "Pilih Sentimen",
        options=df["hasil_sentimen"].unique(),
        default=df["hasil_sentimen"].unique()
    )
    # topek
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