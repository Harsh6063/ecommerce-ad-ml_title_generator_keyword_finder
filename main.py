import streamlit as st
from engine import get_output
import pandas as pd

df = pd.read_csv("data/processed/final_dataset.csv")

st.title("🚀 AI Ad Optimization Engine")

category = st.selectbox("Category", df["category"].unique())
brand = st.text_input("Brand Name")

k1 = st.text_input("Keyword 1")
k2 = st.text_input("Keyword 2")

if st.button("Generate 🚀"):

    res = get_output(category, brand, k1, k2)

    st.subheader("🧠 Generated Title")
    st.write(res["title"])

    st.subheader("🔑 Suggested Keywords")
    st.write(res["keywords"])

    st.subheader("⚔️ Competition")
    st.write(res["competition"])

    st.subheader("📊 Score")
    st.write(res["score"])

    st.subheader("💰 Suggested Bid")
    st.write(res["bid"])