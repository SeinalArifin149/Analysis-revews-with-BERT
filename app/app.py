import streamlit as st
import pandas as pd
from conf import conf_page
from load_data import load_data
from sidebar import filter_sidebar
st.sidebar.title("Filter Data")

df=load_data()
filtered_sidebar= filter_sidebar()

st.dataframe(filtered_sidebar)
