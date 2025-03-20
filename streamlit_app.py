import streamlit as st
import pandas as pd
import numpy as np
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
    st.write("Data Visualization")


def dummy_predict(data):
    # Simulasi output probabilitas untuk setiap kelas obesitas
    prob = np.random.dirichlet(np.ones(6), size=1)[0]  # Probabilitas acak dengan total 1
    categories = ["Insufficient Weight", "Normal Weight", "Overweight Level I",
                  "Overweight Level II", "Obesity Type I", "Obesity Type II"]
    
    # Membuat DataFrame hasil prediksi probabilitas
    prob_df = pd.DataFrame([prob], columns=categories)
    
    # Mendapatkan indeks dengan probabilitas tertinggi
    predicted_class = np.argmax(prob)
    
    return prob_df, predicted_class

# **Judul Aplikasi**
st.title("Obesity Prediction App")

# **Input Data Pengguna**
st.header("Input Data")
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 80, 25)
height = st.slider("Height (m)", 1.2, 2.2, 1.7)
weight = st.slider("Weight (kg)", 30, 200, 70)
family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
favc = st.selectbox("Frequent Consumption of High Caloric Food", ["yes", "no"])
fcvc = st.slider("Frequency of Vegetable Consumption", 1, 3, 2)
ncp = st.slider("Number of Main Meals", 1, 4, 3)
caec = st.selectbox("Consumption of Food Between Meals", ["Sometimes", "Frequently", "Always", "No"])

# **Menampilkan Data yang Diinputkan**
input_data = pd.DataFrame([{
    "Gender": gender, "Age": age, "Height": height, "Weight": weight,
    "family_history_with_overweight": family_history,
    "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec
}])

st.write("Data input by user")
st.dataframe(input_data)

# **Prediksi Model**
st.write("Obesity Prediction")
probabilities, predicted_class = dummy_predict(input_data)

# Menampilkan probabilitas setiap kelas
st.dataframe(probabilities)

# Menampilkan hasil prediksi akhir
st.write("The predicted output is: ", predicted_class)
