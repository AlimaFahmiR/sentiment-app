import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from preprocess import preprocess_text
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model & tokenizer & label encoder
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("lstm_model_20000.h5", compile=False)
    return model

@st.cache_resource
def load_tokenizer():
    with open("tokenizer_20000.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return tokenizer

@st.cache_resource
def load_label_encoder():
    with open("label_encoder_20000.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return label_encoder

# Load resources
model = load_model()
tokenizer = load_tokenizer()
label_encoder = load_label_encoder()

MAX_LEN = 30

# Judul aplikasi
st.title("📱 Sentiment Analysis")

# Input teks
user_input = st.text_area("Enter your review")

if st.button("Predict"):
    if user_input.strip() != "":
        # Preprocessing + debug
        tokens, seq, padded = preprocess_text(user_input, tokenizer, MAX_LEN,  return_tokens=True)

        # Prediksi
        probs = model.predict(padded)
        pred_class = np.argmax(probs, axis=1)[0]
        pred_label = label_encoder.inverse_transform([pred_class])[0]

        preprocessed_text = ' '.join(tokens)

        st.subheader(f"✅ Hasil Prediksi: {pred_label}")

    else:
        st.warning("⚠️ Tolong masukkan teks ulasan terlebih dahulu.")
