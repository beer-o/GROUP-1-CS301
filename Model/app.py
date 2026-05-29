import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# Configure the web application layout
st.set_page_config(
    page_title="Breast Cancer Diagnostic Panel",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Pickled Random Forest - Cancer Prediction Interface")
st.markdown(
    "Adjust the cellular metrics below. The model will run real-time inference using the pickled model state."
)
st.markdown("""
👥 GROUP MEMBERS
* Abel Archibong - ADS24B00176Y
* Anago Michael - ABS24A00030Y 
* Assitan Camara - ABS24A00015Y  
---
""", unsafe_allow_html=True)                      

# Define the exact feature column list required by the model in sequence
FEATURE_KEYS = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst", "fractal_dimension_worst"
]

# Initialize fallback dictionary variables to completely prevent NameError
model = None
defaults = {}

# Load the pickled model artifact safely
@st.cache_resource
def load_pickled_pipeline():
    # Looks precisely inside the folder where app.py lives
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "cancer_model.pkl")
    
    # Fallback to look one level up if it's structured in a sibling directory
    if not os.path.exists(model_path):
        base_dir = os.path.dirname(current_dir)
        model_path = os.path.join(base_dir, "Model", "cancer_model.pkl")

    with open(model_path, "rb") as file:
        payload = pickle.load(file)
    return payload

try:
    pipeline = load_pickled_pipeline()
    model = pipeline["model"]
    defaults = pipeline["defaults"]
except Exception as e:
    st.error(f"🚨 **Could not find or load 'cancer_model.pkl'**")
    st.info("💡 **How to fix this:** Please run your `train.py` script first to generate the pickled model artifact before starting the Streamlit server.")
    st.code("python c:/Users/Haqi/Desktop/CS301_Demo/Model/train.py", language="bash")
    st.stop() 


# Segment the 30 features into category tabs to keep the user interface tidy
tab1, tab2, tab3 = st.tabs(
    ["Mean Metrics", "Standard Error Metrics", "Worst/Extreme Metrics"]
)

input_features = {}

# Layout Section 1: Means
with tab1:
    st.subheader("Mean Attributes of Cell Nuclei")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_mean"] = st.number_input(
            "Radius (Mean)", value=defaults.get("radius_mean", 14.12)
        )
        input_features["texture_mean"] = st.number_input(
            "Texture (Mean)", value=defaults.get("texture_mean", 19.28)
        )
        input_features["perimeter_mean"] = st.number_input(
            "Perimeter (Mean)", value=defaults.get("perimeter_mean", 91.96)
        )
    with c2:
        input_features["area_mean"] = st.number_input(
            "Area (Mean)", value=defaults.get("area_mean", 654.88)
        )
        input_features["smoothness_mean"] = st.number_input(
            "Smoothness (Mean)", value=defaults.get("smoothness_mean", 0.096)
        )
        input_features["compactness_mean"] = st.number_input(
            "Compactness (Mean)", value=defaults.get("compactness_mean", 0.104)
        )
    with c3:
        input_features["concavity_mean"] = st.number_input(
            "Concavity (Mean)", value=defaults.get("concavity_mean", 0.088)
        )
        input_features["concave points_mean"] = st.number_input(
            "Concave Points (Mean)", value=defaults.get("concave points_mean", 0.048)
        )
        input_features["symmetry_mean"] = st.number_input(
            "Symmetry (Mean)", value=defaults.get("symmetry_mean", 0.181)
        )
        input_features["fractal_dimension_mean"] = st.number_input(
            "Fractal Dimension (Mean)", value=defaults.get("fractal_dimension_mean", 0.062)
        )

# Layout Section 2: Standard Errors
with tab2:
    st.subheader("Standard Error (SE) Attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_se"] = st.number_input(
            "Radius SE", value=defaults.get("radius_se", 0.405)
        )
        input_features["texture_se"] = st.number_input(
            "Texture SE", value=defaults.get("texture_se", 1.216)
        )
        input_features["perimeter_se"] = st.number_input(
            "Perimeter SE", value=defaults.get("perimeter_se", 2.866)
        )
    with c2:
        input_features["area_se"] = st.number_input(
            "Area SE", value=defaults.get("area_se", 40.337)
        )
        input_features["smoothness_se"] = st.number_input(
            "Smoothness SE", value=defaults.get("smoothness_se", 0.007)
        )
        input_features["compactness_se"] = st.number_input(
            "Compactness SE", value=defaults.get("compactness_se", 0.025)
        )
    with c3:
        input_features["concavity_se"] = st.number_input(
            "Concavity SE", value=defaults.get("concavity_se", 0.031)
        )
        input_features["concave points_se"] = st.number_input(
            "Concave Points SE", value=defaults.get("concave points_se", 0.011)
        )
        input_features["symmetry_se"] = st.number_input(
            "Symmetry SE", value=defaults.get("symmetry_se", 0.020)
        )
        input_features["fractal_dimension_se"] = st.number_input(
            "Fractal Dimension SE", value=defaults.get("fractal_dimension_se", 0.003)
        )

# Layout Section 3: Worst Metrics
with tab3:
    st.subheader("Worst/Largest Recorded Attributes")
    c1, c2, c3 = st.columns(3)
    with c1:
        input_features["radius_worst"] = st.number_input(
            "Radius (Worst)", value=defaults.get("radius_worst", 16.269)
        )
        input_features["texture_worst"] = st.number_input(
            "Texture (Worst)", value=defaults.get("texture_worst", 25.677)
        )
        input_features["perimeter_worst"] = st.number_input(
            "Perimeter (Worst)", value=defaults.get("perimeter_worst", 107.261)
        )
    with c2:
        input_features["area_worst"] = st.number_input(
            "Area (Worst)", value=defaults.get("area_worst", 880.583)
        )
        input_features["smoothness_worst"] = st.number_input(
            "Smoothness (Worst)", value=defaults.get("smoothness_worst", 0.132)
        )
        input_features["compactness_worst"] = st.number_input(
            "Compactness (Worst)", value=defaults.get("compactness_worst", 0.254)
        )
    with c3:
        input_features["concavity_worst"] = st.number_input(
            "Concavity (Worst)", value=defaults.get("concavity_worst", 0.272)
        )
        input_features["concave points_worst"] = st.number_input(
            "Concave Points (Worst)", value=defaults.get("concave points_worst", 0.114)
        )
        input_features["symmetry_worst"] = st.number_input(
            "Symmetry (Worst)", value=defaults.get("symmetry_worst", 0.290)
        )
        input_features["fractal_dimension_worst"] = st.number_input(
            "Fractal Dimension (Worst)", value=defaults.get("fractal_dimension_worst", 0.083)
        )

st.write("---")

# Execution block triggered upon selection
if st.button("Generate Classification Diagnostic", type="primary", use_container_width=True):
    # Enforce precise feature extraction order using hardcoded schema layout
    feature_vector = np.array([input_features[key] for key in FEATURE_KEYS]).reshape(1, -1)

    # Query the unpickled model instance
    prediction = model.predict(feature_vector)[0]
    probabilities = model.predict_proba(feature_vector)[0]

    st.subheader("Diagnostic Results")

    # Present diagnostic evaluation output with proper UX alert styling
    if prediction == 1:
        st.error(
            f"🚨 **Diagnostic Result: Malignant** — Classification Confidence: {probabilities[1]*100:.2f}%"
        )
    else:
        st.success(
            f"✅ **Diagnostic Result: Benign** — Classification Confidence: {probabilities[0]*100:.2f}%"
        )

    # Render a probability chart
    chart_data = pd.DataFrame(
        {"Probability State (%)": probabilities * 100}, index=["Benign", "Malignant"]
    )
    st.bar_chart(chart_data)