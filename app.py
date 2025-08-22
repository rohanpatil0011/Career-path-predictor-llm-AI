import streamlit as st
import pandas as pd
import joblib
from groq import Groq
from PIL import Image

# Load pre-trained ML model (you’ll train & save separately)
@st.cache_resource
def load_model():
    return joblib.load("student_model.pkl")

# Streamlit UI
st.title("🎓 Student Performance Prediction + AI Assistant")

# Sidebar: API key input
api_key = st.sidebar.text_input("Groq API Key", type="password")

# Upload student dataset (CSV)
uploaded_file = st.file_uploader("Upload Student Data (CSV)", type=["csv"])

# Upload image (optional)
uploaded_img = st.file_uploader("Upload Student Image (optional)", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Student Image", use_column_width=True)

# Text input for LLM queries
user_prompt = st.text_area("💬 Ask AI (e.g., 'Why is the student at risk?')")

# Button for prediction
if st.button("🔍 Predict & Explain"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        model = load_model()
        predictions = model.predict(df)
        df["Prediction"] = predictions
        st.write("📊 Predictions:", df)

        # AI Explanation
        if api_key.startswith("gsk_") and user_prompt:
            client = Groq(api_key=api_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "You are an education advisor."},
                    {"role": "user", "content": f"Predictions: {df.to_dict()} \n Question: {user_prompt}"}
                ]
            )
            st.success(response.choices[0].message.content)
        else:
            st.warning("⚠ Please enter a valid Groq API key & prompt for explanation")
    else:
        st.error("⚠ Please upload student dataset first")
