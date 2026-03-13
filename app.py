import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("predictive_maintenance_dataset.csv")

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🔧 Predictive Maintenance Machine Learning Dashboard")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Navigation",
    ["Home","Dataset","EDA Visualization","Prediction"]
)

# HOME PAGE
if menu == "Home":
    st.header("Machine Failure Prediction System")
    st.write("This application predicts machine failure using Machine Learning.")

# DATASET PAGE
elif menu == "Dataset":
    st.header("Dataset Preview")
    st.dataframe(df.head())

    st.subheader("Dataset Shape")
    st.write(df.shape)

    st.subheader("Statistical Summary")
    st.write(df.describe())

# EDA PAGE
elif menu == "EDA Visualization":

    st.header("Exploratory Data Analysis")

    # Histogram
    st.subheader("Temperature Distribution")
    fig = plt.figure()
    sns.histplot(df["Temperature"], kde=True)
    st.pyplot(fig)

    # Boxplot
    st.subheader("RPM Boxplot")
    fig = plt.figure()
    sns.boxplot(x=df["RPM"])
    st.pyplot(fig)

    # Scatter plot
    st.subheader("Temperature vs Vibration")
    fig = plt.figure()
    sns.scatterplot(x=df["Temperature"], y=df["Vibration"])
    st.pyplot(fig)

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

# PREDICTION PAGE
elif menu == "Prediction":

    st.header("Machine Failure Prediction")

    Temperature = st.number_input("Temperature")
    Vibration = st.number_input("Vibration")
    Pressure = st.number_input("Pressure")
    Humidity = st.number_input("Humidity")
    RPM = st.number_input("RPM")
    Voltage = st.number_input("Voltage")
    Current = st.number_input("Current")
    Machine_Age = st.number_input("Machine Age")

    if st.button("Predict Failure"):

        features = np.array([[Temperature,Vibration,Pressure,Humidity,RPM,Voltage,Current,Machine_Age]])

        prediction = model.predict(features)

        if prediction[0] == 1:
            st.error("⚠ Machine Failure Likely")
        else:
            st.success("✅ Machine Working Normally")