import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.preprocessing import MinMaxScaler
import datetime
import qrcode
import io

# Load trained model and selected features
@st.cache_resource
def load_model():
    with open("best_housing_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("best_features.json", "r") as f:
        features = json.load(f)
    return model, features

model, selected_features = load_model()

# Load scaler
@st.cache_resource
def load_scaler():
   df = pd.read_csv("https://drive.google.com/uc?export=download&id=1A5-9qfWlt3EWvjcoZwHOH_ZXrMFazy7D")
    scaler = MinMaxScaler()
    scaler.fit(df[selected_features])
    return scaler

scaler = load_scaler()

# App title
st.title("🏠 House Price Predictor (China 🇨🇳)")
st.markdown("Fill in the details below to estimate the house price (in Chinese Yuan ¥):")

# Input form
user_input = {}

for feature in selected_features:
    if feature == "communityAverage":
        user_input[feature] = st.number_input("Community Average", min_value=0, max_value=100, value=50, step=1)
    elif feature == "tradeTime":
        date = st.date_input("Trade Time", datetime.date(2020, 1, 1))
        user_input[feature] = int(date.strftime("%Y%m%d"))
    elif feature == "constructionTime":
        user_input[feature] = st.number_input("Construction Year", min_value=1950, max_value=2025, value=2010, step=1)
    elif feature in ["livingRoom", "drawingRoom", "bathRoom"]:
        user_input[feature] = st.slider(f"{feature}", min_value=0, max_value=5, value=1)
    elif feature == "fiveYearsProperty":
        user_input[feature] = 1 if st.radio("5 Years Property?", ["Yes", "No"]) == "Yes" else 0
    elif feature.lower().startswith("floor") or "floor" in feature.lower():
        user_input[feature] = 1 if st.radio(f"{feature.replace('_', ' ').capitalize()}?", ["Yes", "No"], key=feature) == "Yes" else 0
    else:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

# Convert to DataFrame and scale
input_df = pd.DataFrame([user_input])
input_scaled = scaler.transform(input_df)

# Predict button
if st.button("Predict Price"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"🏡 Estimated House Price: *¥{prediction:,.2f}*")

# QR Code generation
def generate_qr_code(link):
    qr = qrcode.make(link)
    buf = io.BytesIO()
    qr.save(buf, format="PNG")
    buf.seek(0)
    st.image(buf, caption="📱 Scan this QR Code to try this app on your phone!")

# App QR link
app_link = "https://your-app-link.streamlit.app"  # Replace with actual link after deploying
generate_qr_code(app_link)

