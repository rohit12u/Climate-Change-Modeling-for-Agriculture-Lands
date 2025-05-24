# climate_modeling_india.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import base64
import folium
from streamlit_folium import st_folium
import io
from matplotlib.backends.backend_pdf import PdfPages

# Page setup
st.set_page_config(page_title="Climate Change Modeling - India", layout="wide")

# Background image
st.markdown(
    """
    <style>
    .stApp {
        background: url('https://media.istockphoto.com/id/179057833/photo/view-of-contrasting-landscape.jpg?s=612x612&w=0&k=20&c=yvpj4kNTgc8ysReGfBopqoBfj4p5RlD0ugz74EV6lhU=') no-repeat center center fixed;
        background-size: cover;
        background-blend-mode: multiply;
        background-color: rgba(0, 0, 0, 0.5); /* Darken with black at 50% opacity */
        
    }
    .sidebar .sidebar-content {
        background: rgba(300, 255, 255, 0.8); /* Light sidebar for readability */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar navigation
page = st.sidebar.radio("Navigation", ["🏠 Introduction", "📘 User Manual","🌾 Crop Yield Modeling"])

# Load dataset or upload
@st.cache_data
def load_default_data():
    return pd.read_csv("indian_states_climate_crop_data_2010_2024.csv")

# Coordinates of Indian states
state_coords = {
    "Andhra Pradesh": [15.9129, 79.7400], "Bihar": [25.0961, 85.3131], "Chhattisgarh": [21.2787, 81.8661],
    "Gujarat": [22.2587, 71.1924], "Haryana": [29.0588, 76.0856], "Karnataka": [15.3173, 75.7139],
    "Kerala": [10.8505, 76.2711], "Madhya Pradesh": [22.9734, 78.6569], "Maharashtra": [19.7515, 75.7139],
    "Odisha": [20.9517, 85.0985], "Punjab": [31.1471, 75.3412], "Rajasthan": [27.0238, 74.2179],
    "Tamil Nadu": [11.1271, 78.6569], "Telangana": [18.1124, 79.0193], "Uttar Pradesh": [26.8467, 80.9462],
    "Uttarakhand": [30.0668, 79.0193], "West Bengal": [22.9868, 87.8550]
}

# Introduction Page
if page == "🏠 Introduction":
    st.title("Climate Change Modeling in Agricultural Land")
    st.markdown("""
    This project presents a web-based system to analyze and predict crop yield responses to climate change across various Indian states. Built using Streamlit and machine learning, the application allows users to explore climate data—including temperature, rainfall, humidity, and soil moisture—across multiple years (2010 to 2024).

    By selecting a state, crop, and year range, users can view predictive yield outcomes and trends. Users can upload their datasets or use default synthetic data, and even enter real-time weather inputs to see how predicted yield would change. Interactive charts display how climate features correlate with agricultural productivity, and all insights are exportable for offline use.

    The system aims to assist farmers, researchers, and policymakers in making informed decisions about crop planning under varying environmental conditions. Through data visualization, location-aware maps, and robust predictions, this application bridges the gap between data science and sustainable farming.
    Key Features:
    - Upload or use default dataset
    - Predict yield for selected state, year, and crop
    - Show recommendations
    - Display India map with predicted results
    - Download forecast
    
    
    """)

elif page == "🌾 Crop Yield Modeling":
    st.title("🌿 State-wise Crop Yield Prediction")

    st.header("📥 User Inputs")
    uploaded_file = st.file_uploader("Upload your dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Uploaded custom dataset.")
    else:
        # df = load_default_data()
        st.info("Using default dataset.")

    if 'df' in locals():
        st.sidebar.header("📌 Select Inputs")
        state = st.sidebar.selectbox("Select State", sorted(df['State'].unique()))
        crop = st.sidebar.selectbox("Select Crop", sorted(df['Crop'].unique()))
        year_range = st.sidebar.slider("Select Year Range", min_value=int(df['Year'].min()), max_value=int(df['Year'].max()), value=(2015, 2024))

        filtered = df[(df['State'] == state) & (df['Crop'] == crop) & (df['Year'].between(*year_range))]
        if filtered.empty:
            st.error("No data available for this selection.")
            st.stop()

        st.subheader("📊 Historical Feature Trends Before Prediction")
        features = ["Temperature", "Rainfall", "Humidity", "Soil_Moisture"]
        for feature in features:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=filtered, x="Year", y=feature, marker="o", ax=ax)
            ax.set_title(f"{feature} Trend in {state} for {crop}")
            ax.set_ylabel(feature)
            ax.set_xlabel("Year")
            st.pyplot(fig)

        st.subheader("📊 Crop Yield Based on Climate Parameters")
        pdf_buf = io.BytesIO()
        with PdfPages(pdf_buf) as pdf:
            for feature in features:
                fig, ax = plt.subplots(figsize=(8, 4))
                sns.barplot(data=filtered, x=feature, y="Crop_Yield", ax=ax)
                ax.set_xticklabels([f'{tick:.2f}' for tick in filtered[feature]])
                ax.set_title(f"Crop Yield vs {feature} in {state} for {crop}")
                ax.set_xlabel(feature)
                ax.set_ylabel("Crop Yield (kg/ha)")
                st.pyplot(fig)
                pdf.savefig(fig)
                plt.close(fig)

        pdf_buf.seek(0)

        pdf_buf.seek(0)
        st.download_button(
            label="📄 Download All Plots as PDF",
            data=pdf_buf,
            file_name="crop_yield_vs_climate_factors.pdf",
            mime="application/pdf"
        )

        
        

        st.header("🌾 Future Crop Yield Prediction")
        temp = st.number_input("Enter Temperature (°C)", min_value=0.0, value=30.0)
        rain = st.number_input("Enter Rainfall (mm)", min_value=0.0, value=1000.0)
        hum = st.number_input("Enter Humidity (%)", min_value=0.0, value=50.0)
        input_df = pd.DataFrame({"Temperature": [temp], "Rainfall": [rain], "Humidity": [hum], "Soil_Moisture": [30]})

        X = df[features]
        y = df["Crop_Yield"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        predicted_yield = model.predict(input_df)[0]
        st.subheader(f"Predicted Crop Yield: {predicted_yield/1000:.2f} tons/hectare")

        coords = state_coords.get(state, [22.59, 78.96])
        st.subheader("🗺️ State Location Map")
        m = folium.Map(location=coords, zoom_start=6)
        folium.Marker(location=coords, tooltip=f"{state} - {crop}: {predicted_yield:.2f} kg/ha").add_to(m)
        st_folium(m, use_container_width=True, height=500)

        st.subheader("⬇️ Download Prediction")
        result_df = pd.DataFrame({
            "State": [state], "Year Range": [f"{year_range[0]}-{year_range[1]}"], "Crop": [crop], "Predicted_Yield (kg/ha)": [predicted_yield]
        })
        st.download_button("Download as CSV", result_df.to_csv(index=False), file_name="prediction_result.csv")

        


elif page == "📘 User Manual":
    st.title("📘 User Manual")
    st.markdown("""
    This project is designed to help visualize and analyze the effects of climate change on crop yield across Indian states.

    ### What This Tool Does:
    - Allows users to upload their own datasets or use default data.
    - Lets you select a state, crop, and year range to analyze yield patterns.
    - Accepts manual input of climate parameters to predict future yield using machine learning.
    - Displays accurate map locations for each state with crop-specific yield predictions.
    - Provides visual bar charts comparing yield based on temperature, rainfall, humidity, and soil moisture.
    - All charts can be exported to PDF and results downloaded as CSV files.

    ### How to Use:
    1. Upload your dataset or use the default.
    2. Choose your state, crop, and year range.
    3. Enter climate values (temperature, rainfall, humidity).
    4. View predictions, visual insights, and maps.
    5. Download results using buttons provided at the end.

    This app is user-friendly and built with Streamlit. It provides deep insights into agriculture's response to climate using visual and predictive intelligence.
    """)

st.markdown("---")
st.markdown("**Developed by [RohitSen(rsen95759@gmail.com)]** | Climate Change Modeling for Agricultural Lands 🌾")

