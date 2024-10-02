# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 21:36:51 2024

@author: Dami
"""

import numpy as np
import pandas as pd
import pickle
import streamlit

filename = "C:/Users/Dami/Downloads/trained_model_and_encodings.sav"
loaded_data = pickle.load(open(filename, 'rb'))


loaded_model = loaded_data['model']  # SVR model
loaded_model_encoding = loaded_data['model_encoding']  # Model encodings (from target encoding)
loaded_make_encoding = loaded_data['make_encoding']  # Make encodings (from target encoding)

# Function to encode 'Model' and 'Make' based on target encoding
def encode_input(model, make):
    model_encoded = loaded_model_encoding.get(model, 0)  # Default to 0 if not found
    make_encoded = loaded_make_encoding.get(make, 0)  # Default to 0 if not found
    return model_encoded, make_encoded

# Streamlit app UI
st.title("Electric Vehicle Price Prediction")

# Input fields for the user
model_year = st.number_input("Model Year", min_value=2000, max_value=2024, value=2021)
electric_range = st.number_input("Electric Range (miles)", min_value=50, max_value=500, value=150)
base_msrp = st.number_input("Base MSRP ($)", min_value=10000, max_value=200000, value=50000)
age_of_vehicle = st.number_input("Age of Vehicle (years)", min_value=0, max_value=20, value=3)

# Dropdown for `Model` and `Make` with readable labels
model = st.selectbox("Vehicle Model", options=list(loaded_model_encoding.keys()))
make = st.selectbox("Vehicle Make", options=list(loaded_make_encoding.keys()))

# **CAFV Eligibility** Dropdown with Descriptive Labels
cafv_eligibility = st.selectbox("CAFV Eligibility", options={
    "Eligible": 1,
    "Not Eligible": 0
})

# **EV Type** Dropdown with Descriptive Labels
ev_type = st.selectbox("EV Type", options={
    "Battery Electric Vehicle (BEV)": 1,
    "Plug-in Hybrid Electric Vehicle (PHEV)": 0
})

# Encode the selected `Model` and `Make`
model_encoded, make_encoded = encode_input(model, make)

# Prepare input data for prediction
input_data = pd.DataFrame({
    'Model Year': [model_year],
    'Electric Range': [electric_range],
    'Base MSRP': [base_msrp],
    'Age of Vehicle': [age_of_vehicle],
    '(CAFV)_Eligibility_encoded': [cafv_eligibility],  # Encoded value from dropdown
    'EV_Type_encoded': [ev_type],  # Encoded value from dropdown
    'Model_encoded': [model_encoded],  # Target encoded value
    'Make_encoded': [make_encoded]   # Target encoded value
})

# Display the input data
st.write("Input Data:", input_data)

# Predict the price
if st.button("Predict Price"):
    predicted_price = loaded_model.predict(input_data)
    st.write(f"Predicted Vehicle Price: ${predicted_price[0]:,.2f} (in 1,000s)")