import streamlit as st
import pickle
import numpy as np
import pandas as pd


# Load model and features
with open("villa_price_prediction.pkl", "rb") as f:
    model = pickle.load(f)

with open("model_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

st.title("Villa Price Prediction App 💰🏡")

# Define inputs
bathrooms = st.number_input("Bathrooms", min_value=0, step=1)
bedrooms = st.number_input("Bedrooms", min_value=0, step=1)
beds = st.number_input("Beds", min_value=0, step=1)
latitude = st.number_input("Latitude", format="%.6f")
longitude = st.number_input("Longitude", format="%.6f")

# Binary features example
has_wifi = st.checkbox("Has WiFi?")
has_ac = st.checkbox("Has Air Conditioning?")
# Add more amenities as needed...

# City options (you can extract actual city names from model_columns)
cities = [col.replace("city_", "") for col in model_columns if col.startswith("city_")]
selected_city = st.selectbox("Select City", cities)

# One-hot encode city
city_data = {f"city_{city}": 1 if city == selected_city else 0 for city in cities}

# Base input
input_data = {
    "bathrooms": bathrooms,
    "bedrooms": bedrooms,
    "beds": beds,
    "latitude": latitude,
    "longitude": longitude,
    "has_wireless_internet": int(has_wifi),
    "has_air_conditioning": int(has_ac),
    # Add more binary flags if needed
}

# Combine all features
full_input = {**input_data, **city_data}

# Ensure all model features are present
for col in model_columns:
    if col not in full_input:
        full_input[col] = 0  # fill missing with 0

# Predict
if st.button("Predict Price"):
    X = pd.DataFrame([full_input])[model_columns]
    y_pred = model.predict(X)
    price = np.exp(y_pred[0])  # convert log_price back to actual
    st.success(f"Estimated Villa Price: ₹{price:,.2f} per night")
