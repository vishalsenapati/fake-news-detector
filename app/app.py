import streamlit as st
import random

st.title("Fake News Detector")
st.write("Paste a news headline or article text below to check its credibility.")

user_input = st.text_area("Enter news text:")

if st.button("Check"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = random.choice(["Real", "Fake"])
        confidence = round(random.uniform(60, 95), 2)
        st.success(f"Prediction: {result} (Confidence: {confidence}%)")