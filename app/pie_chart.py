import streamlit as st
import matplotlib.pyplot as plt

def sentimen_pie_chart(filtered_df):

    sentimen_count = filtered_df["hasil_sentimen"].value_counts()

    if sentimen_count.empty:
        return

    fig, ax = plt.subplots(figsize=(3.5,3.5))

    ax.pie(
        sentimen_count,
        labels=sentimen_count.index,
        autopct='%1.1f%%',
        radius=0.75,
        textprops={'fontsize':8}
    )

    ax.set_title(
        "Distribusi Sentimen",
        fontsize=10
    )

    plt.tight_layout()

    st.pyplot(fig)