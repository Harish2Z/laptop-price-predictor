import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================
# LOAD MODEL + COLUMNS
# =============================

model = pickle.load(open("final_model.pkl", "rb"))
model_columns = pickle.load(open("model_columns.pkl", "rb"))

st.set_page_config(page_title="Laptop Price Predictor", layout="wide")

st.title("💻 Laptop Price Prediction System")
st.markdown("Professional Machine Learning Model for Laptop Price Estimation")

# =============================
# INPUT SECTIONS
# =============================

st.header("Basic Specifications")

col1, col2, col3 = st.columns(3)

with col1:
    brand = st.selectbox(
        "Brand",
        ["HP", "Dell", "Lenovo", "Acer", "Asus", "MSI", "Apple"]
    )

with col2:
    cpu_brand = st.selectbox(
        "CPU Brand",
        ["Intel", "AMD", "Apple"]
    )

with col3:
    cpu_tier = st.selectbox(
        "CPU Tier",
        ["i3", "i5", "i7", "i9",
         "ryzen3", "ryzen5", "ryzen7", "ryzen9",
         "m1", "m2"]
    )

st.header("Performance Specifications")

col4, col5, col6 = st.columns(3)

with col4:
    cpu_generation = st.slider("CPU Generation", 5, 14, 12)

with col5:
    ram = st.selectbox("RAM (GB)", [4, 8, 16, 32, 64])

with col6:
    storage = st.selectbox("Storage (GB)", [256, 512, 1024, 2048])

col7, col8, col9 = st.columns(3)

with col7:
    gpu_type = st.selectbox("GPU Type", ["Integrated", "Dedicated"])

with col8:
    gpu_vram = st.selectbox("GPU VRAM (GB)", [0, 2, 4, 6, 8])

with col9:
    os_type = st.selectbox("Operating System", ["Windows", "Mac"])

st.header("Display & Other")

col10, col11, col12 = st.columns(3)

with col10:
    display_size = st.selectbox("Display Size (inches)",
                                [13.3, 14.0, 15.6, 16.0, 17.3])

with col11:
    resolution_map = {
        "720p": 1280 * 720,
        "1080p": 1920 * 1080,
        "1440p": 2560 * 1440,
        "4K": 3840 * 2160
    }

    resolution_choice = st.selectbox(
        "Resolution",
        list(resolution_map.keys())
    )

    resolution_pixels = resolution_map[resolution_choice]

with col12:
    warranty = st.selectbox("Warranty (Years)", [0, 1, 2, 3])

# =============================
# PREDICTION
# =============================

if st.button("Predict Price"):

    input_dict = {
        'cpu_generation': cpu_generation,
        'gpu_vram': gpu_vram,
        'Ram_gb': ram,
        'Storage_gb': storage,
        'display_size': display_size,
        'resolution_pixels': resolution_pixels,
        'warranty': warranty,
        'brand_' + brand: 1,
        'cpu_brand_' + cpu_brand: 1,
        'cpu_tier_' + cpu_tier: 1,
        'gpu_type_' + gpu_type: 1,
        'os_type_' + os_type: 1
    }

    input_df = pd.DataFrame(columns=model_columns)
    input_df.loc[0] = 0

    for key in input_dict:
        if key in input_df.columns:
            input_df.at[0, key] = input_dict[key]

    pred_log = model.predict(input_df)[0]
    predicted_price = np.expm1(pred_log)

    st.success(f"Estimated Laptop Price: ₹ {int(predicted_price):,}")
