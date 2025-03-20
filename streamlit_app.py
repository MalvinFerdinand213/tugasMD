import streamlit as st
import pandas as pd

# Set page title
st.title("Machine Learning App")

# Subtitle
st.markdown("### This app will predict your obesity level!")

# Load sample data (replace with your actual dataset)
data = {
    "Gender": ["Female", "Female", "Male", "Male", "Male", "Male", "Female", "Male", "Male", "Male"],
    "Age": [21, 21, 23, 27, 22, 29, 23, 24, 21, 22],
    "Height": [1.62, 1.52, 1.80, 1.87, 1.78, 1.62, 1.53, 1.78, 1.78, 1.72],
    "Weight": [64, 56, 76, 87, 89.8, 53, 55, 85, 73, 68],
    "family_history_with_overweight": ["yes", "yes", "yes", "no", "no", "yes", "yes", "yes", "yes", "yes"],
    "FAVC": ["no", "no", "no", "no", "no", "yes", "no", "yes", "yes", "yes"],
    "FCVC": [2, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    "NCP": [3, 3, 3, 2, 3, 3, 3, 3, 3, 3],
    "CAEC": ["Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes", "Sometimes"]
}

df = pd.DataFrame(data)

# Expander to show raw data
with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(df)
