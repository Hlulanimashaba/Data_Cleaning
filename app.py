import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.impute import SimpleImputer
import base64

# Function to generate a download link for a file
def get_download_link(file_path):
    with open(file_path, 'rb') as file:
        contents = file.read()
    base64_encoded = base64.b64encode(contents).decode()
    href = f'<a href="data:file/csv;base64,{base64_encoded}" download="{file_path}">Click here to download</a>'
    return href

# Function to detect rows with anomalies using Isolation Forest
def detect_anomalies(data):
    # Create a copy of the data to avoid modifying the original
    data_copy = data.select_dtypes(include=[np.number]).copy()
    
    # Fill missing values with the mean of the column
    imputer = SimpleImputer(strategy='mean')
    data_imputed = imputer.fit_transform(data_copy)
    
    # Initialize Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    
    # Fit Isolation Forest and predict anomalies
    preds = clf.fit_predict(data_imputed)
    anomaly_mask = preds == -1
    
    return anomaly_mask

# Function to clean data using various strategies for imputation
def clean_data(data):
    # Create a copy of the data to avoid modifying the original
    data_copy = data.copy()
    
    # Detect anomalies using Isolation Forest
    anomaly_mask = detect_anomalies(data_copy)
    
    # Separate numerical, categorical, and datetime columns
    numerical_cols = data_copy.select_dtypes(include=[np.number]).columns
    categorical_cols = data_copy.select_dtypes(include=['object']).columns
    datetime_cols = data_copy.select_dtypes(include=['datetime64']).columns
    
    # Impute categorical columns using the most frequent value
    if len(categorical_cols) > 0:
        cat_imputer = SimpleImputer(strategy='most_frequent')
        data_copy[categorical_cols] = cat_imputer.fit_transform(data_copy[categorical_cols])
    
    # Impute numerical columns using RandomForestRegressor
    for column in numerical_cols:
        if data_copy[column].isnull().sum() > 0:
            X = data_copy.loc[~data_copy[column].isnull(), data_copy.columns != column]
            y = data_copy.loc[~data_copy[column].isnull(), column]

            # Determine if the column is predominantly integer or float
            num_int = np.sum(y.apply(lambda x: float(x).is_integer()))
            num_float = len(y) - num_int
            predominant_dtype = int if num_int > num_float else float
            
            if len(X) > 0 and not y.isnull().any():
                try:
                    num_imputer = RandomForestRegressor(n_estimators=100, random_state=42)
                    num_imputer.fit(X, y)
                    predicted_values = num_imputer.predict(data_copy.loc[data_copy[column].isnull(), data_copy.columns != column])
                    
                    # Cast the predicted values to the predominant data type
                    if predominant_dtype is int:
                        predicted_values = np.round(predicted_values).astype(int)
                    else:
                        predicted_values = predicted_values.astype(float)
                    
                    data_copy.loc[data_copy[column].isnull(), column] = predicted_values
                except ValueError as e:
                    st.warning(f"Error imputing column '{column}': {str(e)}")
            else:
                st.warning(f"No samples available to train imputer for column '{column}'.")
    
    # Handle datetime columns separately
    for column in datetime_cols:
        if data_copy[column].isnull().sum() > 0:
            # Fill NaNs with the most frequent datetime value
            most_frequent_date = data_copy[column].mode()[0]
            data_copy[column] = data_copy[column].fillna(most_frequent_date)
    
    # Impute string (object) columns using the most frequent value
    if len(categorical_cols) > 0:
        string_imputer = SimpleImputer(strategy='most_frequent')
        data_copy[categorical_cols] = string_imputer.fit_transform(data_copy[categorical_cols])
    
    # Ensure data types are preserved for numerical columns
    for column in numerical_cols:
        if pd.api.types.is_integer_dtype(data[column]):
            data_copy[column] = data_copy[column].astype(pd.Int64Dtype())
        elif pd.api.types.is_float_dtype(data[column]):
            data_copy[column] = data_copy[column].astype(float)
    
    return data_copy

# Title of the app
st.title("ML Algorithm to Detect and Clean Missing Data")

# Step 1: Data Loading
st.header("Step 1: Data Loading")
uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Determine the file type and load the data accordingly
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith('.xlsx'):
        data = pd.read_excel(uploaded_file)
    
    st.write("Data Loaded Successfully:")
    st.write(data.head())

    # Step 2: Data Inspection
    st.header("Step 2: Data Inspection")
    st.write("Summary Statistics:")
    st.write(data.describe(include='all'))
    
    st.write("Missing Values Count:")
    st.write(data.isnull().sum())

    # Step 3: Detecting and Cleaning Missing Data
    st.header("Step 3: Detecting and Cleaning Missing Data")
    
    # Clean data
    cleaned_data = clean_data(data)
    
    # Display cleaned data
    st.subheader("Cleaned Data:")
    st.write(cleaned_data.head())
    
    # Save Cleaned Data
    cleaned_data_filename = "cleaned_data_imputed.csv"
    cleaned_data.to_csv(cleaned_data_filename, index=False)
    st.success(f"Cleaned data saved as {cleaned_data_filename}")
    
    # Provide download link to the user
    st.markdown(get_download_link(cleaned_data_filename), unsafe_allow_html=True)

else:
    st.write("Please upload a CSV or Excel file to proceed.")

# Footer
st.markdown("<p style='text-align:center;'>Developed by Mashaba Hlulani Charles</p>", unsafe_allow_html=True)
