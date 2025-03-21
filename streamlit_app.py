import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

class ObesityModel:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoders = {}
        self.scaler = StandardScaler()

    def preprocess(self, df, training=True):
        """ Melakukan encoding pada data kategorikal dan normalisasi pada data numerik """
        categorical_features = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 
                                'SMOKE', 'SCC', 'CALC', 'MTRANS']
        numeric_features = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']

        df = df.copy()

        # Encoding kategori
        for col in categorical_features:
            if training:
                self.label_encoders[col] = LabelEncoder()
                df[col] = self.label_encoders[col].fit_transform(df[col])
            else:
                df[col] = self.label_encoders[col].transform(df[col])

        # Normalisasi numerik
        if training:
            df[numeric_features] = self.scaler.fit_transform(df[numeric_features])
        else:
            df[numeric_features] = self.scaler.transform(df[numeric_features])

        return df

    def train(self, X, y):
        X = self.preprocess(X, training=True)
        self.model.fit(X, y)

    def predict(self, input_data):
        input_data = self.preprocess(input_data, training=False)
        probabilities = self.model.predict_proba(input_data)
        predicted_class = self.model.predict(input_data)
        return probabilities, predicted_class

    def save_model(self, filename="obesity_model.pkl"):
        with open(filename, "wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load_model(filename="obesity_model.pkl"):
        with open(filename, "rb") as file:
            return pickle.load(file)

# Load dataset
df = pd.read_csv("dataset.csv")  # Ganti dengan path dataset kamu
X = df.drop(columns=["NObeyesdad"])  # Fitur
y = df["NObeyesdad"]  # Label

# Inisialisasi dan training model
obesity_model = ObesityModel()
obesity_model.train(X, y)
obesity_model.save_model()

# Streamlit UI
st.title("Obesity Prediction App")
st.subheader("Masukkan Data Anda")

# Input dari user
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", 10, 100, 25)
height = st.slider("Height (m)", 1.0, 2.5, 1.7)
weight = st.slider("Weight (kg)", 30, 200, 70)
family_history = st.selectbox("Family History of Overweight", ["yes", "no"])
favc = st.selectbox("Frequent Consumption of High Caloric Food", ["yes", "no"])
fcvc = st.slider("Vegetable Consumption Frequency (1-3)", 1, 3, 2)
ncp = st.slider("Number of Main Meals (1-4)", 1, 4, 3)
caec = st.selectbox("Consumption of Food Between Meals", ["no", "Sometimes", "Frequently", "Always"])
smoke = st.selectbox("Do you smoke?", ["yes", "no"])
ch2o = st.slider("Daily Water Consumption (1-3)", 1, 3, 2)
scc = st.selectbox("Calories Consumption Monitoring", ["yes", "no"])
faf = st.slider("Physical Activity Frequency (0-3)", 0, 3, 1)
tue = st.slider("Time using Technology Devices (0-2)", 0, 2, 1)
calc = st.selectbox("Consumption of Alcohol", ["no", "Sometimes", "Frequently", "Always"])
mtrans = st.selectbox("Transportation used", ["Automobile", "Motorbike", "Bike", "Public_Transport", "Walking"])

# Buat DataFrame dari input user
input_data = pd.DataFrame([[gender, age, height, weight, family_history, favc, fcvc, ncp, caec, smoke, 
                            ch2o, scc, faf, tue, calc, mtrans]],
                          columns=X.columns)

# Load model dan lakukan prediksi
obesity_model = ObesityModel.load_model()
probabilities, predicted_class = obesity_model.predict(input_data)

# Tampilkan hasil prediksi
st.subheader("Obesity Prediction")
st.write("Probabilities per class:")
st.write(pd.DataFrame(probabilities, columns=obesity_model.model.classes_))
st.write(f"**Predicted Class:** {predicted_class[0]}")  
