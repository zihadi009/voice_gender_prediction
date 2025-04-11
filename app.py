import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import random

# Page Config
st.set_page_config(page_title="Voice Gender Classifier", layout="centered")

# Load the model
filename = 'voice_model.pickle'
loaded_model = pickle.load(open(filename, 'rb'))

# Feature labels
feature_info = {
    'meanfreq': 'Mean Frequency',
    'sd': 'Standard Deviation',
    'IQR': 'Interquartile Range',
    'kurt': 'Kurtosis',
    'sp.ent': 'Spectral Entropy',
    'mode': 'Mode',
    'meanfun': 'Mean Fundamental Frequency',
    'maxfun': 'Maximum Fundamental Frequency',
    'meandom': 'Mean of Dominant Frequency',
    'modindx': 'Modulation Index'
}

# Sidebar Info
with st.sidebar:
    st.markdown("## 🧐 About")
    st.write("This app predicts the **gender of a speaker** based on voice features using a pre-trained ML model.")
    st.markdown("---")
    st.markdown("👨‍💻 *Created by MD. Musfiqur Rahman and Md. Masrafi Al Amin*")
    st.markdown("📁 Model: `voice_model.pickle`")

# Title + Header
st.markdown("<h1 style='text-align: center; color: #4e8cff;'>🎤 Voice Gender Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Enter the values below to predict if the voice is male or female.</p>", unsafe_allow_html=True)
st.markdown("---")

# Reset logic (before widgets)
if "reset_triggered" not in st.session_state:
    st.session_state.reset_triggered = False

if st.button("🔄 Reset All Fields"):
    st.session_state.reset_triggered = True
    st.rerun()

if st.session_state.reset_triggered:
    for feature in feature_info:
        st.session_state[feature] = 0.0
    st.session_state.reset_triggered = False

# Example autofill input with random values
if st.button("🎲 Try Example Input"):
    for key in feature_info:
        if key in ['meanfreq', 'sd', 'IQR', 'sp.ent', 'mode', 'meanfun', 'maxfun', 'meandom', 'modindx']:
            st.session_state[key] = round(random.uniform(0.0, 0.5), 5)
        elif key == 'kurt':
            st.session_state[key] = round(random.uniform(1.0, 5.0), 5)
    st.rerun()

# Input fields
st.markdown("### 🎧 Input Acoustic Features")
user_inputs = {}
for feature, label in feature_info.items():
    user_inputs[feature] = st.number_input(
        f"{label}",
        step=0.01,
        format="%.5f",
        key=feature
    )

# Predict Button
if st.button("🚀 Predict Gender"):
    input_df = pd.DataFrame([user_inputs])
    prediction = loaded_model.predict(input_df)[0]
    probability = loaded_model.predict_proba(input_df)[0]
    confidence = round(max(probability) * 100, 2)
    gender = "Male" if prediction == 1 else "Female"
    color = "#4e8cff" if gender == "Male" else "#ff69b4"

    st.markdown(f"<h2 style='text-align: center; color: {color};'>🧬 Predicted Gender: {gender}</h2>", unsafe_allow_html=True)
    st.info(f"🔍 Confidence: {confidence}%")

    # Show probability as bar chart
    st.markdown("#### 🔬 Model Probability")
    labels = ['Female', 'Male']
    plt.bar(labels, probability, color=['pink', 'skyblue'])
    st.pyplot(plt)

    # Feature importance (if available)
    if hasattr(loaded_model, "feature_importances_"):
        st.markdown("#### 📈 Feature Importance")
        importance = loaded_model.feature_importances_
        sorted_idx = importance.argsort()
        plt.figure()
        plt.barh([list(feature_info.values())[i] for i in sorted_idx], importance[sorted_idx], color='#4e8cff')
        st.pyplot(plt)

# CSV Upload
st.markdown("---")
st.markdown("### 📂 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    predictions = loaded_model.predict(df)
    df['Predicted Gender'] = ['Male' if p == 1 else 'Female' for p in predictions]
    st.dataframe(df)
    st.download_button("Download Results", df.to_csv(index=False), file_name="gender_predictions.csv")

# Footer
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div style='text-align: center; color: gray;'>Made with ❤️ using Streamlit</div>", unsafe_allow_html=True)