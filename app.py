import streamlit as st
import pandas as pd
import joblib
from PIL import Image
import google.generativeai as genai
import os
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# Gemini API setup (use Streamlit secrets for safety)
# ------------------------------
if "GEMINI_API_KEY" not in st.secrets:
    st.sidebar.error("⚠ Please add GEMINI_API_KEY in .streamlit/secrets.toml")
else:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")

def ask_ai(prompt):
    try:
        response = model_gemini.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"⚠ Error: {e}"

# ------------------------------
# Load local ML model if exists
# ------------------------------
@st.cache_resource
def load_model():
    if os.path.exists("student_model.pkl"):
        return joblib.load("student_model.pkl")
    else:
        return None

# ------------------------------
# Streamlit UI
# ------------------------------
st.title("🎓 Career Path Prediction + AI Assistant (Gemini)")

st.markdown(
    """
    Upload student data containing **marks, hobbies, and certificates**.  
    The system will predict the **most probable career path** using a trained ML model.  
    Then, Gemini AI will **explain why this career path is suitable** and answer your questions.  
    """
)

# Upload student dataset (CSV)
uploaded_file = st.file_uploader("📂 Upload Student Data (CSV)", type=["csv"])

# Upload student image (optional)
uploaded_img = st.file_uploader("🖼️ Upload Student Image (optional)", type=["jpg", "jpeg", "png"])

if uploaded_img is not None:
    image = Image.open(uploaded_img)
    st.image(image, caption="Uploaded Student Image", use_column_width=True)

# Text input for LLM queries
user_prompt = st.text_area("💬 Ask AI (e.g., 'Why is this student suited for engineering?')")

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("🔍 Predict Career Path & Explain"):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Load ML model
        model_local = load_model()

        if model_local:
            try:
                # Handle categorical encoding if needed
                if "Hobby" in df.columns and "Certificates" in df.columns:
                    le_hobby = LabelEncoder()
                    le_cert = LabelEncoder()
                    df["Hobby"] = le_hobby.fit_transform(df["Hobby"].astype(str))
                    df["Certificates"] = le_cert.fit_transform(df["Certificates"].astype(str))

                # Predict careers
                predictions = model_local.predict(df)
                df["Career Path Prediction"] = predictions
            except Exception as e:
                df["Career Path Prediction"] = f"⚠ Prediction error: {e}"
        else:
            df["Career Path Prediction"] = "⚠ No ML model found (AI-only mode)"

        st.subheader("📊 Predictions")
        st.write(df)

        # AI Explanation
        if user_prompt:
            result = ask_ai(
                f"Student data: {df.to_dict()} \n"
                f"Question: {user_prompt} \n"
                f"Explain based on marks, hobbies, and certificates which career path is probable."
            )
            st.subheader("🧠 AI Explanation")
            st.success(result)
        else:
            st.warning("⚠ Please enter a question for AI explanation")
    else:
        st.error("⚠ Please upload student dataset first")
