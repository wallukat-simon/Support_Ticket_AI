import streamlit as st
import os
import requests

API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

st.set_page_config(page_title="Ticket Classifier", layout="centered")

st.title("Support Ticket Classifier")

# UI Layout
st.subheader("Model Selection")
model_choice = st.radio(
    "Choose the model for prediction:",
    ["DistilBERT (Transformer)", "TF-IDF + ML (CPU)"],
    horizontal=True
)

st.subheader("Input")
text = st.text_area(
    "Enter a support ticket:",
    height=180,
    placeholder="Example: I cannot log into my account and need help resetting my password."
)

# Prediction
if st.button("Predict", use_container_width=True):
    if text.strip() == "":
        st.warning("Please enter some text.")
    else:
        model_type = "bert" if model_choice == "DistilBERT (Transformer)" else "ml"

        payload = {
            "text": text,
            "model_type": model_type
        }

        try:
            response = requests.post(API_URL, json=payload, timeout=10)
            result = response.json()

            st.success("Prediction")
            st.markdown(f"**Category:** `{result['category']}`")
            st.markdown(f"**Confidence:** `{result['confidence']:.2f}`")
            st.caption(f"Model used: {result['model']}")

        except Exception as e:
            st.error("API request failed")
            st.exception(e)


st.markdown("---")
st.caption("NLP Project: Classical ML vs Transformer-based Models")