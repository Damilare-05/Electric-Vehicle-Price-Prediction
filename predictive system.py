# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle

# Load the model and encodings from the .sav file
filename = "C:/Users/Dami/Downloads/trained_model_and_encodings.sav"
loaded_data = pickle.load(open(filename, 'rb'))

# Extract the model and the encodings
loaded_model = loaded_data['model']  # SVR model
loaded_model_encoding = loaded_data['model_encoding']  # Model encodings
loaded_make_encoding = loaded_data['make_encoding']  # Make encodings

# Let's assume we have new data for prediction
new_model = "MODEL 3"
new_make = "TESLA"
new_model_year = 2022
new_electric_range = 350  
new_base_msrp = 35000     
new_vehicle_age = 2        
new_cafv_eligibility = 1   
new_ev_type = 1           

# Apply the encoding using the loaded encoding mappings
encoded_model = loaded_model_encoding.get(new_model, 0)  
encoded_make = loaded_make_encoding.get(new_make, 0)     

# Create the input data list with all required features
new_data = [
    new_model_year,         # 'Model Year'
    new_electric_range,     # 'Electric Range'
    new_base_msrp,          # 'Base MSRP'
    new_vehicle_age,        # 'Age of Vehicle'
    new_cafv_eligibility,   # '(CAFV)_Eligibility_encoded'
    new_ev_type,            # 'EV_Type_encoded'
    encoded_model,          # 'Model_encoded'
    encoded_make            # 'Make_encoded'
]

# Make a prediction with the loaded model
predicted_price = loaded_model.predict([new_data])

print(f"Predicted price: ${predicted_price[0]}k")