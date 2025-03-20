import pandas as pd
import streamlit as st

df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") 
st.write(df)
