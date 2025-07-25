import streamlit as st
from model_helper import predict
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_license_plate_text(image_path):
    ocr = PaddleOCR(lang='en')
    results = ocr.ocr(image_path, cls=True)
    license_text = " ".join([res[1][0] for line in results for res in line if res[1][1] > 0.5])
    return license_text.strip()

st.title("Vehicle Damage Detection with License Plate Verification")

uploaded_file = st.file_uploader("Upload the file", type=["jpg", "png"])

allowed_plates = {"MH20EE7602", "TJG990GP", "TS04FA","Minnesota 999999","HR76A2085"} # Example license numbers

if uploaded_file:
    image_path = "temp_file.jpg"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.image(uploaded_file, caption="Uploaded File", use_container_width=True)
    
    # Extract license plate number
    license_number = extract_license_plate_text(image_path)
    st.write(f"Extracted License Plate: {license_number}")
    
    if license_number in allowed_plates:
        st.success("License plate is verified. Proceeding to damage detection...")
        prediction = predict(image_path)
        st.info(f"Predicted Damage Class: {prediction}")
    else:
        st.error("License plate not found in database. Access Denied.")