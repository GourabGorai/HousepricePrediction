import streamlit as st
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Function to add a row to the CSV file
def add_row_to_csv(file_path, data):
    with open(file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(data)

# Load the existing dataset
file_path = 'Housing.csv'
data = pd.read_csv(file_path)

# Split the data into features (X) and target (y)
X = data.drop('price', axis=1)
y = data['price']

# Define preprocessing steps
categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating',
                        'airconditioning', 'prefarea', 'furnishingstatus']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),  # Handle missing values
    ('onehot', OneHotEncoder(drop='first'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ], remainder='passthrough')  # Keep non-categorical features

# Create pipeline with preprocessing and linear regression
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Train the model
model.fit(X, y)

# Get user input for new data
st.title('House Price Prediction')
area = st.number_input("Enter the area in cm^2: ")
bedrooms = st.number_input("Enter number of bedrooms: ")
bathrooms = st.number_input("Enter number of bathrooms: ")
stories = st.number_input("Enter number of stories: ")
mainroad = st.radio("Mainroad available or not:", ('yes', 'no'))
guestroom = st.radio("Guestroom available or not:", ('yes', 'no'))
basement = st.radio("Basement available or not:", ('yes', 'no'))
hotwaterheating = st.radio("Hotwater heating available or not:", ('yes', 'no'))
airconditioning = st.radio("AC available or not:", ('yes', 'no'))
parking = st.number_input("Number of parking: ")
prefarea = st.radio("Preferable area available or not:", ('yes', 'no'))
furnishingstatus = st.selectbox("Select furnishing status:", ('furnished', 'semi-furnished', 'unfurnished'))

# Create new data
new_data = pd.DataFrame({
    'area': [area],
    'bedrooms': [bedrooms],
    'bathrooms': [bathrooms],
    'stories': [stories],
    'mainroad': [mainroad],
    'guestroom': [guestroom],
    'basement': [basement],
    'hotwaterheating': [hotwaterheating],
    'airconditioning': [airconditioning],
    'parking': [parking],
    'prefarea': [prefarea],
    'furnishingstatus': [furnishingstatus]
})

# Predict price
predicted_price = model.predict(new_data)
st.write("Predicted price:", predicted_price[0])

# Add predicted data to CSV file
new_data_list = [predicted_price[0], area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus]
add_row_to_csv(file_path, new_data_list)

# Plotting existing data and predicted data
plt.figure(figsize=(20, 10))

# Plot existing data
plt.scatter(data['area'], data['price'], color='blue', label='Existing Data')

# Plot predicted data
plt.scatter(area, predicted_price, color='red', label='Predicted Data')

plt.title('House Price Prediction')
plt.xlabel('Area')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

st.pyplot(plt)