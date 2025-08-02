import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from io import BytesIO
import base64

# ---------------------------
# App Setup
# ---------------------------
st.set_page_config(page_title="Swasthya Sathi", layout="wide")
model = joblib.load("blood cancer prediction/models/blood_cancer_model.pkl")
logo = Image.open("blood cancer prediction/assets/blood-test.png")

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

banner_base64 = get_base64_image("blood cancer prediction/assets/Doctors-bro.png")

HEALTHY_RANGES = {
    "Male": {
        "WBC": (4000, 11000),
        "RBC": (4.7, 6.1),
        "Platelets": (150000, 450000),
        "Hemoglobin": (13.8, 17.2),
        "Blasts": (0, 5)
    },
    "Female": {
        "WBC": (4000, 11000),
        "RBC": (4.2, 5.4),
        "Platelets": (150000, 450000),
        "Hemoglobin": (12.1, 15.1),
        "Blasts": (0, 5)
    }
}

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = []

# ---------------------------
# Styling
# ---------------------------
st.markdown("""
<style>
.stApp { background-color: #111827; color: white; }
[data-testid="stSidebar"] { background-color: #1f2937; }
h1 { font-size: 30px !important; }
h2, h3 { font-size: 22px !important; }
p, label, div, span { font-size: 16px !important; color: white !important; }
.result-card {
    font-size: 16px;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Banner Image
# ---------------------------
st.markdown(
    f"""
    <div style='text-align: center;'>
        <img src="data:image/png;base64,{banner_base64}" width="400"/>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("<h1 style='text-align:center;'>🩺 Swasthya Sathi: Your Health Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>AI-powered blood cancer prediction with clinical support</p>", unsafe_allow_html=True)

# ---------------------------
# Sidebar
# ---------------------------
with st.sidebar:
    st.image(logo, width=150)
    st.title("🩺 Swasthya Sathi")
    page = st.radio("Go to", ["Make Prediction", "Patient Statistics", "Model Info"])

# ---------------------------
# Prediction Page
# ---------------------------
if page == "Make Prediction":
    st.header("🧾 Enter Patient Details")

    # Gender (full width)
    gender = st.selectbox("Gender", ["Male", "Female"], key="gender_input")

    # Age and WBC Count
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, key="age_input")
    with col2:
        wbc = st.number_input("WBC Count (/µL)", step=0.01, key="wbc_input")

    # RBC and Platelet Count
    col3, col4 = st.columns(2)
    with col3:
        rbc = st.number_input("RBC Count (million/µL)", step=0.01, key="rbc_input")
    with col4:
        platelets = st.number_input("Platelet Count (/µL)", step=1, key="platelets_input")

    # Hemoglobin and Blasts
    col5, col6 = st.columns(2)
    with col5:
        hemoglobin = st.number_input("Hemoglobin Level (g/dL)", step=0.1, key="hemoglobin_input")
    with col6:
        blasts = st.number_input("Bone Marrow Blasts (%)", step=1, key="blasts_input")

    if st.button("Predict 🧪"):
        input_data = [[age, wbc, rbc, platelets, hemoglobin, blasts]]
        ref = HEALTHY_RANGES[gender]
        violations = sum([
            not (ref["WBC"][0] <= wbc <= ref["WBC"][1]),
            not (ref["RBC"][0] <= rbc <= ref["RBC"][1]),
            not (ref["Platelets"][0] <= platelets <= ref["Platelets"][1]),
            not (ref["Hemoglobin"][0] <= hemoglobin <= ref["Hemoglobin"][1]),
            not (ref["Blasts"][0] <= blasts <= ref["Blasts"][1])
        ])

        result = "Positive" if violations > 0 else ("Positive" if model.predict(input_data)[0] == 1 else "Negative")

        # ✅ Custom background & border for result card
        bg_color = "#ff4d6d" if result == "Positive" else "#32e0c4"
        border_color = "#b30000" if result == "Positive" else "#0f5132"
        emoji = "🟥" if result == "Positive" else "🟩"

        st.markdown(
            f"""
            <div class='result-card' style='background-color:{bg_color}; border-left: 10px solid {border_color};
                 padding: 15px; border-radius: 10px; margin-top: 15px;'>
                <h3>{emoji} Prediction Result: Leukemia {result}</h3>
                <p><strong>Violations Detected:</strong> {violations}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Save patient record
        patient = {
            "Gender": gender, "Age": age, "WBC": wbc, "RBC": rbc,
            "Platelets": platelets, "Hemoglobin": hemoglobin, "Blasts": blasts,
            "Violations": violations, "Prediction": result,
            "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M")
        }
        st.session_state.patient_data.append(patient)

        # PDF Report
        buffer = BytesIO()
        c = canvas.Canvas(buffer)
        c.setFont("Helvetica-Bold", 16)
        c.drawString(150, 800, "Swasthya Sathi - Patient Report")
        c.setFont("Helvetica", 12)
        y = 760
        for k, v in patient.items():
            c.drawString(70, y, f"{k}: {v}")
            y -= 20
        c.save()
        buffer.seek(0)
        st.download_button("📄 Download Patient Report", data=buffer,
                           file_name="swasthya_report.pdf", mime="application/pdf")

# ---------------------------
# Patient Statistics Page
# ---------------------------
elif page == "Patient Statistics":
    st.header("📊 Reference Ranges (Static)")
    data = []
    for gender, values in HEALTHY_RANGES.items():
        for param, (minv, maxv) in values.items():
            data.append({"Gender": gender, "Parameter": param, "Min": minv, "Max": maxv})
    df = pd.DataFrame(data)
    st.dataframe(df)

# ---------------------------
# Model Info Page
# ---------------------------
elif page == "Model Info":
    st.header("🧠 Model & Clinical Information")
    st.markdown("""
    - Model: Random Forest Classifier (~84% accuracy)  
    - Features: Age, Gender, WBC, RBC, Platelets, Hemoglobin, Blasts  
    - If any value violates normal range, prediction is overridden to Positive.
    """)
