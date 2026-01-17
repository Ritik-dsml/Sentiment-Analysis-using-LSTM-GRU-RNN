import streamlit as st
import numpy as np
import pickle
import re
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# =====================================================
# Page Configuration
# =====================================================
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="📊",
    layout="centered"
)

# =====================================================
# Text Preprocessing (EXACT MATCH)
# =====================================================
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

# =====================================================
# Load Model & Tokenizer
# =====================================================
@st.cache_resource
def load_artifacts():
    model = load_model("rnn_model.h5", compile=False)
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_artifacts()

# 🔴 MUST MATCH TRAINING
MAX_LEN = 50   # from your notebook

# LabelEncoder alphabetical mapping
LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}

# =====================================================
# UI Header
# =====================================================
st.markdown(
    """
    <h1 style="text-align:center;">🔍 Sentiment Analysis System</h1>
    <p style="text-align:center; color: gray;">
    Deep Learning based Text Sentiment Classifier (RNN / LSTM / GRU)
    </p>
    """,
    unsafe_allow_html=True
)

st.divider()

# =====================================================
# Input Section
# =====================================================
st.subheader("📝 Enter Text for Sentiment Analysis")

user_text = st.text_area(
    "Type or paste your text below:",
    height=150,
    placeholder="that boy is having nice haircut"
)

analyze_btn = st.button("Analyze Sentiment 🚀")

# =====================================================
# Prediction Logic (IDENTICAL TO CLI)
# =====================================================
def predict_sentiment(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN)  # padding='pre'
    
    prediction = model.predict(padded, verbose=0)
    pred_idx = np.argmax(prediction, axis=1)[0]
    
    return LABEL_MAP[pred_idx]

# =====================================================
# Output Section
# =====================================================
if analyze_btn:
    if user_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing sentiment..."):
            result = predict_sentiment(user_text)

        color_map = {
            "Positive": "green",
            "Negative": "red",
            "Neutral": "orange"
        }

        emoji_map = {
            "Positive": "😊",
            "Negative": "😞",
            "Neutral": "😐"
        }

        st.subheader("📊 Analysis Result")

        st.markdown(
            f"""
            <h2 style="color:{color_map[result]}; text-align:center;">
                {result} {emoji_map[result]}
            </h2>
            """,
            unsafe_allow_html=True
        )

        st.divider()

# =====================================================
# Sidebar
# =====================================================
with st.sidebar:
    st.header("📌 About This Project")
    st.write(
        """
        - **Model:** SimpleRNN / LSTM / GRU  
        - **Framework:** TensorFlow & Keras  
        - **Frontend:** Streamlit  
        - **Dataset:** Twitter Sentiment Dataset  
        - **Classes:** Positive / Neutral / Negative  
        """
    )

    st.markdown("---")

    st.subheader("👨‍💻 Created By")
    st.write("**Ritik Kumar**")
    st.write("Data Analyst | AI & ML Enthusiast")

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/ritik-kumar-mlai) | "
        "[GitHub](https://github.com/Ritik-dsml)",
        unsafe_allow_html=True
    )
