import streamlit as st
import pandas as pd
import csv
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Function to convert DataFrame to CSV
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

# Function to validate the uploaded file
def validate_file(uploaded_file):
    """Checks if the uploaded file is a CSV with the required columns."""
    try:
        df = pd.read_csv(uploaded_file, nrows=10)
        if all(col in df.columns for col in required_columns):
            return True, df
        else:
            return False, f"Missing required columns: {', '.join(set(required_columns) - set(df.columns))}"
    except Exception as e:
        return False, f"Error reading file: {str(e)}"

# Required columns in the CSV file
required_columns = ['price', 'area', 'bedrooms', 'bathrooms', 'stories', 'mainroad', 'guestroom', 'basement',
                    'hotwaterheating', 'airconditioning', 'parking', 'prefarea', 'furnishingstatus']

st.write(
    "Welcome to HOUSE PRICE PREDICTION software. Upload only .csv files and must have the following columns:- price, area, bedrooms, bathrooms, stories, mainroad, guestroom, basement, hotwaterheating, airconditioning, parking, prefarea, furnishingstatus")
st.write("Created by:-")
st.write("Name:- Gourab Gorai | Reg. no:- 223231010051 | University Roll no.:- 32301222016")

file = st.file_uploader("Choose a CSV file", type="csv")

if file is not None:
    is_valid, message = validate_file(file)
    if is_valid:
        st.success("File uploaded successfully!")

        # Reset the buffer to the beginning after validation
        file.seek(0)

        # Load the existing dataset
        data = pd.read_csv(file)

        # Initialize session state to store data
        if 'data' not in st.session_state:
            st.session_state.data = data
            st.session_state.selling_prices = []
            st.session_state.predicted_prices = []

        # Sort the data by the 'area' column
        data = st.session_state.data.sort_values(by='area')

        # Split the data into features (X) and target (y)
        X = data.drop('price', axis=1)
        y = data['price']

        # Define preprocessing steps
        categorical_features = ['mainroad', 'guestroom', 'basement', 'hotwaterheating', 'airconditioning', 'prefarea',
                                'furnishingstatus']
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
        area = st.number_input("Enter the area in m^2: ")
        bedrooms = st.number_input("Enter number of bedrooms:", step=1, format="%d")
        bathrooms = st.number_input("Enter number of bathrooms:", step=1, format="%d")
        stories = st.number_input("Enter number of stories:", step=1, format="%d")
        parking = st.number_input("Number of parking:", step=1, format="%d")

        mainroad = st.radio("Mainroad available or not:", ('yes', 'no'))
        guestroom = st.radio("Guestroom available or not:", ('yes', 'no'))
        basement = st.radio("Basement available or not:", ('yes', 'no'))
        hotwaterheating = st.radio("Hotwater heating available or not:", ('yes', 'no'))
        airconditioning = st.radio("AC available or not:", ('yes', 'no'))
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
        st.write("Predicted price: {:.2f}".format(predicted_price[0]))

        sp = st.number_input("Enter the selling price: ")

        if st.button('Submit'):
            # Add predicted data to DataFrame
            new_data['price'] = sp

            # Add new row to existing data DataFrame using pd.concat
            st.session_state.data = pd.concat([st.session_state.data, new_data], ignore_index=True)

            # Save updated data to CSV file
            updated_csv = convert_df_to_csv(st.session_state.data)

            # Show the updated data
            st.write("Updated Data:")
            st.dataframe(st.session_state.data)

            # Provide a download button for the updated CSV file
            st.download_button(
                label="Download updated file",
                data=updated_csv,
                file_name="updated_data.csv",
                mime="text/csv"
            )

            # Plotting existing data and predicted data
            plt.figure(figsize=(20, 10))

            # Plot existing data
            plt.scatter(st.session_state.data['area'], st.session_state.data['price'], color='blue',
                        label='Existing Data')

            # Plot predicted data
            plt.scatter(area, predicted_price, color='red', label='Predicted Data')

            plt.title('House Price Prediction')
            plt.xlabel('Area')
            plt.ylabel('Price')
            plt.legend()
            plt.grid(True)

            st.pyplot(plt)

            # Update lists
            st.session_state.selling_prices.append(sp)
            st.session_state.predicted_prices.append(predicted_price[0])

            # Calculate accuracy if we have 3 data points
            if len(st.session_state.selling_prices) == 3:
                r2 = r2_score(st.session_state.selling_prices, st.session_state.predicted_prices)
                accuracy_percentage = r2 * 100
                st.write(f"Accuracy based on the last 3 entries: {accuracy_percentage:.2f}%")

                # Reset lists
                st.session_state.selling_prices = []
                st.session_state.predicted_prices = []

    else:
        st.error(message)
        st.write("Please upload a valid CSV file with the required columns.")
else:
    st.info("Upload a CSV file to proceed.")
