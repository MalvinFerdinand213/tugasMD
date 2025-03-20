import streamlit as st
import pandas as pd
import plotly.express as px

st.title("Machine Learning App")
st.info("This app will predict your obesity level!")

data = pd.read_csv("ObesityDataSet_raw_and_data_sinthetic.csv") 
df = pd.DataFrame(data)

# Expander to show raw data
with st.expander("Data"):
    st.write("This is a raw data")
    st.dataframe(df)

#Data Visualilzation

with st.expander("Data Visualization"):
    st.write("### Data Visualization")

    # Membuat scatter plot
    fig = px.scatter(df, x="Height", y="Weight", color="Obesity_Type",
                     title="Height vs Weight Distribution",
                     labels={"Height": "Height", "Weight": "Weight"},
                     color_discrete_map={
                         "Insufficient_Weight": "blue",
                         "Normal_Weight": "green",
                         "Obesity_Type_I": "red",
                         "Obesity_Type_II": "orange"
                     })

    st.plotly_chart(fig)
