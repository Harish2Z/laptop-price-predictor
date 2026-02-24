import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and columns
model = pickle.load(open("final_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("💻 Laptop Price Prediction System")
st.markdown("Enter laptop specifications to estimate price.")

# -----------------------------
# INPUT FIELDS
# -----------------------------

brand = st.selectbox("Brand", [
    "HP", "Dell", "Lenovo", "Acer", "Asus", "Apple", "MSI", "Other"
])

cpu_brand = st.selectbox("CPU Brand", ["Intel", "AMD", "Apple"])

cpu_series = st.selectbox("CPU Series", [
    "i3", "i5", "i7", "i9",
    "Ryzen 3", "Ryzen 5", "Ryzen 7", "Ryzen 9",
    "M1", "M2", "Other"
])

cpu_generation = st.number_input("CPU Generation (e.g., 11, 12, 13)", min_value=0, max_value=14, value=12)

gpu_type = st.selectbox("GPU Type", ["Integrated", "Dedicated"])

ram_gb = st.selectbox("RAM (GB)", [4, 8, 12, 16, 24, 32, 64])

ram_type = st.selectbox("RAM Type", ["DDR3", "DDR4", "DDR5", "LPDDR4", "LPDDR5"])

storage_gb = st.selectbox("Storage (GB)", [256, 512, 1024, 2048])

storage_type = st.selectbox("Storage Type", ["HDD", "SSD"])

display_size = st.selectbox("Display Size (inches)", [13.3, 14.0, 15.6, 16.0, 17.3])

resolution_category = st.selectbox("Resolution", ["720p", "1080p", "2K", "4K"])

warranty = st.selectbox("Warranty (Years)", [0, 1, 2, 3])

os_type = st.selectbox("Operating System", ["Windows", "Mac"])

# -----------------------------
# PREDICTION
# -----------------------------

if st.button("Predict Price"):

    input_dict = {
        "brand": brand,
        "cpu_brand": cpu_brand,
        "cpu_series": cpu_series,
        "cpu_generation": cpu_generation,
        "gpu_type": gpu_type,
        "Ram_gb": ram_gb,
        "ram_type": ram_type,
        "Storage_gb": storage_gb,
        "storage_type": storage_type,
        "display_size": display_size,
        "resolution_category": resolution_category,
        "warranty": warranty,
        "os_type": os_type
    }

    input_df = pd.DataFrame([input_dict])

    input_df = pd.get_dummies(input_df, drop_first=True)

    for col in model_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[model_columns]

    prediction_log = model.predict(input_df)
    prediction = np.expm1(prediction_log)[0]

    st.success(f"💰 Estimated Price: ₹ {int(prediction):,}")

