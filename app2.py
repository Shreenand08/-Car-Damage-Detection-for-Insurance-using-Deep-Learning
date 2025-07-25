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

# Step 1: Upload car image with license plate
st.header("Step 1: Upload Car Image with License Plate")
uploaded_license_file = st.file_uploader("Upload the image containing the license plate", type=["jpg", "png"], key="license")

  # Example license numbers
allowed_plates = {"MH20EE7602", "TJG990GP", "TS04FA","Minnesota 999999","HR76A2085"}
license_verified = False

if uploaded_license_file:
    license_image_path = "license_temp.jpg"
    with open(license_image_path, "wb") as f:
        f.write(uploaded_license_file.getbuffer())
        
    st.image(uploaded_license_file, caption="Uploaded License Plate Image", use_container_width=True)
    
    # Extract license plate number
    license_number = extract_license_plate_text(license_image_path)
    st.write(f"Extracted License Plate: {license_number}")
    
    if license_number in allowed_plates:
        st.success("License plate is verified. Please proceed to upload the damaged car image.")
        license_verified = True
    else:
        st.error("License plate not found in database. Access Denied.")

# Step 2: Upload damaged car image
if license_verified:
    st.header("Step 2: Upload Damaged Car Image")
    uploaded_damaged_file = st.file_uploader("Upload the damaged car image", type=["jpg", "png"], key="damaged")
    
    if uploaded_damaged_file:
        damaged_image_path = "damaged_temp.jpg"
        with open(damaged_image_path, "wb") as f:
            f.write(uploaded_damaged_file.getbuffer())
        
        st.image(uploaded_damaged_file, caption="Uploaded Damaged Car Image", use_container_width=True)
        
        # Perform damage detection
        prediction = predict(damaged_image_path)
        st.info(f"Predicted Damage Class: {prediction}")
