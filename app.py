import streamlit as st
st.set_page_config(
   page_title="Klasifikasi Komentar Video Youtube",
   page_icon=":mortar_board:",
   layout="centered",
   initial_sidebar_state="auto",
)
import pandas as pd
from halaman import Metode,Crawling,pre_processing,Labelling, fea_extraction, Modelling
page_names_to_funcs = {
    "Metode/Algoritma" : Metode.app,
    "Crawling"  : Crawling.app,
    "Labelling" : Labelling.app,
    "Pre Processing" : pre_processing.app,
    "Feature Extraction" : fea_extraction.app,
    "Modelling" : Modelling.app
}

demo_name = st.sidebar.selectbox("halaman", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()