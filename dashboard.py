import streamlit as st
import requests

st.title("AI‑Detector Cloud Dashboard")

BASE_URL = st.secrets.get("BASE_URL", "http://localhost:10000")

mode = st.radio("Detection Mode:", ("RoBERTa", "DetectGPT"))
input_type = st.radio("Input Type:", ("Text", "Document file"))

if input_type == "Text":
    text = st.text_area("Text to analyze", height=200)
else:
    file = st.file_uploader("Upload a .txt, .pdf or .docx file")

if st.button("Analyze"):
    if input_type == "Text":
        url = f"{BASE_URL}/{'detect' if mode == 'RoBERTa' else 'detectgpt'}"
        response = requests.post(url, json={"text": text})
    else:
        url = f"{BASE_URL}/upload"
        files = {"file": (file.name, file, file.type)}
        data = {"method": mode.lower()}
        response = requests.post(url, files=files, data=data)
    if response.ok:
        st.json(response.json())
    else:
        st.error(f"Request failed: {response.status_code}")
