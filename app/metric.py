import streamlit as st

def show_metric(filtered_df):
    col1,col2,col3 = st.columns(3)

    with col1:
        st.metric(
            "Jumlah Review",
            filtered_df.shape[0]
        )
    with col2:
        st.metric(
            "Rata Rata Rating",
            round(filtered_df["stars"].mean(), 2)
        )
    with col3:
        st.metric(
            "Jumlah topic",
            filtered_df["nama_topik"].nunique()
        )
    return col1,col2,col3