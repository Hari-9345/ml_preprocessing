import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


st.set_page_config(
    page_title="ML Preprocessing Dashboard",
    layout="wide"
)

st.title("AI Data Preprocessing Dashboard")

st.sidebar.header("Navigation")

menu = st.sidebar.selectbox(
    "Select Module",
    [
        "Upload Dataset",
        "Data Overview",
        "Missing Value Handling",
        "Encoding",
        "Feature Scaling",
        "Feature Engineering",
        "Visualization",
        "Train Test Split"
    ]
)

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    if menu == "Upload Dataset":

        st.subheader("Dataset Preview")

        st.dataframe(data)

        col1, col2, col3 = st.columns(3)

        col1.metric("Rows", data.shape[0])
        col2.metric("Columns", data.shape[1])
        col3.metric("Missing Values", data.isnull().sum().sum())

    # ---------------------------
    # Data Overview
    # ---------------------------

    elif menu == "Data Overview":

        st.subheader("Dataset Information")

        st.write(data.describe())

        st.subheader("Data Types")

        st.write(data.dtypes)

    # ---------------------------
    # Missing Values
    # ---------------------------

    elif menu == "Missing Value Handling":

        st.subheader("Missing Values")

        st.write(data.isnull().sum())

        strategy = st.selectbox(
            "Select Imputation Strategy",
            ["mean", "median", "most_frequent"]
        )

        if st.button("Apply Imputation"):

            imputer = SimpleImputer(strategy=strategy)

            numeric_cols = data.select_dtypes(include=np.number).columns

            data[numeric_cols] = imputer.fit_transform(data[numeric_cols])

            st.success("Missing values handled successfully")

            st.write(data)

    # ---------------------------
    # Encoding
    # ---------------------------

    elif menu == "Encoding":

        st.subheader("Categorical Encoding")

        categorical = data.select_dtypes(include="object").columns

        column = st.selectbox("Select Column", categorical)

        if st.button("Label Encode"):

            le = LabelEncoder()

            data[column] = le.fit_transform(data[column])

            st.write(data)

        if st.button("One Hot Encode"):

            data = pd.get_dummies(data, columns=[column])

            st.write(data)

    # ---------------------------
    # Feature Scaling
    # ---------------------------

    elif menu == "Feature Scaling":

        st.subheader("Scaling Methods")

        numeric = data.select_dtypes(include=np.number).columns

        column = st.selectbox("Select Numeric Column", numeric)

        scaler_type = st.selectbox(
            "Select Scaling",
            ["StandardScaler", "MinMaxScaler", "Normalizer"]
        )

        if st.button("Apply Scaling"):

            if scaler_type == "StandardScaler":
                scaler = StandardScaler()

            elif scaler_type == "MinMaxScaler":
                scaler = MinMaxScaler()

            else:
                scaler = Normalizer()

            data[[column]] = scaler.fit_transform(data[[column]])

            st.write(data)

    # ---------------------------
    # Feature Engineering
    # ---------------------------

    elif menu == "Feature Engineering":

        st.subheader("Polynomial Features")

        numeric = data.select_dtypes(include=np.number)

        degree = st.slider("Select Polynomial Degree", 2, 4)

        if st.button("Generate Features"):

            poly = PolynomialFeatures(degree=degree)

            poly_features = poly.fit_transform(numeric)

            st.write(poly_features)

    # ---------------------------
    # Visualization
    # ---------------------------

    elif menu == "Visualization":

        st.subheader("Interactive Data Visualization")

        column = st.selectbox("Select Column", data.columns)

        chart = st.selectbox(
            "Select Chart",
            ["Histogram", "Box Plot", "Scatter Plot"]
        )

        if chart == "Histogram":

            fig = px.histogram(data, x=column)

            st.plotly_chart(fig)

        elif chart == "Box Plot":

            fig = px.box(data, y=column)

            st.plotly_chart(fig)

        elif chart == "Scatter Plot":

            x = st.selectbox("X Axis", data.columns)

            y = st.selectbox("Y Axis", data.columns)

            fig = px.scatter(data, x=x, y=y)

            st.plotly_chart(fig)

    # ---------------------------
    # Train Test Split
    # ---------------------------

    elif menu == "Train Test Split":

        st.subheader("Split Dataset")

        target = st.selectbox("Select Target Column", data.columns)

        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)

        if st.button("Split Data"):

            X = data.drop(target, axis=1)
            y = data[target]

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )

            st.success("Dataset Split Successful")

            st.write("Train Shape:", X_train.shape)
            st.write("Test Shape:", X_test.shape)