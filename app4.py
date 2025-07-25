import streamlit as st
from model_helper import predict
from paddleocr import PaddleOCR
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load Pretrained ResNet50 Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer
model.to(device)
model.eval()

# Define Image Transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Extract deep features using ResNet50
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.view(-1)  # Flatten to 1D vector

# Compute cosine similarity between two feature vectors
def cosine_similarity(features1, features2):
    return F.cosine_similarity(features1.unsqueeze(0), features2.unsqueeze(0)).item()

# Extract license plate text using PaddleOCR
def extract_license_plate_text(image_path):
    ocr = PaddleOCR(lang='en')
    results = ocr.ocr(image_path, cls=True)
    license_text = " ".join([res[1][0] for line in results for res in line if res[1][1] > 0.5])
    return license_text.strip()

st.title("Vehicle Damage Detection with License Plate Verification")

# Step 1: Upload car image with license plate
st.header("Step 1: Upload Car Image with License Plate")
uploaded_license_file = st.file_uploader("Upload the image containing the license plate", type=["jpg", "png"], key="license")

allowed_plates = {"MH20EE7602", "TJG990GP", "TS04FA","Minnesota 999999","HR76A2085","Encar"}  # Example license numbers
license_verified = False
license_image_path = "license_temp.jpg"

if uploaded_license_file:
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
    damaged_image_path = "damaged_temp.jpg"
    
    if uploaded_damaged_file:
        with open(damaged_image_path, "wb") as f:
            f.write(uploaded_damaged_file.getbuffer())
        
        st.image(uploaded_damaged_file, caption="Uploaded Damaged Car Image", use_container_width=True)
        
        # Extract deep features
        features_license = extract_features(license_image_path)
        features_damaged = extract_features(damaged_image_path)
        
        # Compute similarity
        similarity_score = cosine_similarity(features_license, features_damaged)
        #st.write(f"Deep Learning Similarity Score: {similarity_score:.2f}")
        
        if similarity_score > 0.80:  # Threshold for matching
            # Perform damage detection
            prediction = predict(damaged_image_path)
            st.info(f"Predicted Damage Class: {prediction}")
        else:
            st.error("Uploaded damaged car image does not match the original car. Claim rejected.")
