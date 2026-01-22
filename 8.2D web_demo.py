# This is the code file I created for the web demo application of predicting Melbourne housing prices
import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import datetime

# Web app title
st.title("Melbourne housing price predictor")
st.write("Input property features below to predict the sale price")

# User inputs
col1, col2 = st.columns(2)

with col1:
    suburb = st.selectbox("Suburb", ["Burwood", "Camberwell", "Doncaster"])
    prop_type = st.selectbox("Property type", ["House", "Townhouse", "Unit"])
    parking_area = st.selectbox("Parking area", ["Indoor", "Outdoor stall", "Parkade", "Underground", "Parking pad"])
    building_area = st.number_input("Building area (m²)", 10.0, 1000.0, 150.0)
    landsize = st.number_input("Land size (m²)", 10.0, 2000.0, 500.0)
    year_built = st.slider("Year built", 1900, 2025, 1990)

with col2:
    bedroom = st.slider("Number of bedrooms", 1, 6, 3)
    bathroom = st.slider("Number of bathrooms", 1, 4, 2)
    latitude = st.number_input("Latitude", -38.5, -37.0, -37.8)
    longitude = st.number_input("Longitude", 144.5, 146.0, 145.0)
    selected_date = st.date_input("Sale date",
                                  value=datetime.date.today(),  # default: today
                                  min_value=datetime.date(2000, 1, 1),
                                  max_value=datetime.date(2030, 12, 31))

# Extract year, month, day
year = selected_date.year
month = selected_date.month
day = selected_date.day
# change names for user to the names used by dataset
if prop_type == "House":
    prop_type = "h"
elif prop_type == "Townhouse":
    prop_type = "t"
else:
    prop_type = "u"

# the row of user input
data_input = pd.DataFrame({'Suburb': [suburb],
                           'Type': [prop_type],
                           'Date': [selected_date],
                           'Bedroom': [bedroom],
                           'Bathroom': [bathroom],
                           'Landsize': [landsize],
                           'BuildingArea': [building_area],
                           'YearBuilt': [year_built],
                           'Latitude': [latitude],
                           'Longtitude': [longitude],
                           'ParkingArea': [parking_area],
                           'Price': [999], # doesn't matter
                           'Year': [year],
                           'Month': [month],
                           'Day': [day],
                           'BuildingDensity': [building_area / landsize if landsize > 0 else 0]})

# Add input as first row
data = pd.read_csv("prepared_data.csv")
data = pd.concat([data_input, data], ignore_index=True)

# Select categorical columns to encode
categorical_cols = ['Suburb', 'Type', 'ParkingArea']
# Separate categorical and numerical columns
categorical_data = data[categorical_cols]
numerical_data = data.drop(columns=categorical_cols)
# Initialise OneHotEncoder (drop baseline variable)
encoder = OneHotEncoder(drop='first', sparse_output=False)
# Encode the categorical data
encoded_data = encoder.fit_transform(categorical_data)
# Get new column names from the encoder
encoded_col_names = encoder.get_feature_names_out(categorical_cols)
# Create dataframe from encoded data and column names
encoded_df = pd.DataFrame(encoded_data, columns=encoded_col_names)
# Combine encoded categorical data with original numerical data
data = pd.concat([numerical_data.reset_index(drop=True), encoded_df], axis=1)
# Rename columns
# Rename encoded columns to cleaner names
data = data.rename(columns={'Type_t': 'Type_Townhouse',
                            'Type_u': 'Type_Unit'})

# scaling
numerical_features = ['Bedroom', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt',
                      'Latitude', 'Longtitude', 'BuildingDensity', 'Year', 'Month', 'Day']
scaler = StandardScaler()
data[numerical_features] = scaler.fit_transform(data[numerical_features])

# drop unnecessary columns for predictions
data = data.drop(columns=['Date', 'Price'])
X = data.iloc[[0]]

# Predict
model = joblib.load("house_price_model.pkl")
prediction = model.predict(X)[0]

# Output
st.subheader(f"The estimated price of the property is: ${prediction:,.0f}")