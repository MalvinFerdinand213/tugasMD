import pandas as pd
import streamlit as st

st.info('This app will predict your obesite level!')

def main():
  haha = st.selectbox('Gender')
  df = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") 
  st.write(df)

if __name__ == "__main__":
  main()
