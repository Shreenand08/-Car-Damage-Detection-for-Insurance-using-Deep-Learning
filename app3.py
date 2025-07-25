import streamlit as st
from model_helper import predict
from paddleocr import PaddleOCR
import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def extract_license_plate_text(image_path):
    ocr = PaddleOCR(lang='en')
    results = ocr.ocr(image_path, cls=True)
    license_text = " ".join([res[1][0] for line in results for res in line if res[1][1] > 0.5])
    return license_text.strip()

def load_and_preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (500, 500))  # Resize for consistency
    return image

def compare_images(img1_path, img2_path):
    img1 = load_and_preprocess_image(img1_path)
    img2 = load_and_preprocess_image(img2_path)
    similarity, _ = ssim(img1, img2, full=True)
    return similarity

st.title("Vehicle Damage Detection with License Plate Verification")

# Step 1: Upload car image with license plate
st.header("Step 1: Upload Car Image with License Plate")
uploaded_license_file = st.file_uploader("Upload the image containing the license plate", type=["jpg", "png"], key="license")

allowed_plates = {"MH20EE7602", "TJG990GP", "TS04FA","Minnesota 999999","HR76A2085"} # Example license numbers
license_verified = False
license_image_path = None

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
        
        # Compare images to prevent fraud
        similarity_score = compare_images(license_image_path, damaged_image_path)
        st.write(f"Image Similarity Score: {similarity_score:.2f}")
        
        if similarity_score > 0.5:  # Threshold for similarity
            st.success("Images match. Proceeding with damage detection...")
            prediction = predict(damaged_image_path)
            st.info(f"Predicted Damage Class: {prediction}")
        else:
            st.error("Uploaded damaged car image does not match the original vehicle. Claim denied.")