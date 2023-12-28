import streamlit as st
import tensorflow as tf
from secure_transmission_module import encrypt, decrypt
from adversarial_detection_module import detect_adversarial
from explainability_module import explain_prediction
from medical_condition_classification_module import classify_condition
import cv2
import numpy as np
from tensorflow import keras
from lime import lime_image
import pydicom
import io
import secrets
import csv

# Load the trained models
medical_condition_model = tf.keras.models.load_model('classification_model.h5')
adversarial_detection_model = tf.keras.models.load_model('ad_classification_model.h5')


def read_and_preprocess_image(uploaded_file):
    # Read the image from the uploaded file
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Ensure the image has three channels (convert grayscale to RGB)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    # Resize the image to the required dimensions
    target_size = (224, 224)
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

    # Normalize pixel values to be between 0 and 1
    image = image.astype('float32') / 255.0

    # Expand dimensions to match the model's expected input shape
    image = np.expand_dims(image, axis=0)

    return image


def remove_adversarial_attack(image, perturbation):
    cleaned_image = image - perturbation
    cleaned_image = np.clip(cleaned_image, 0, 255).astype(np.uint8)
    return cleaned_image

def read_and_preprocess_dicom(uploaded_dico):
    # Read the DICOM file from the uploaded file
    dicom_data = pydicom.dcmread(uploaded_dico)

    # Extract pixel data from the DICOM file
    pixel_data = dicom_data.pixel_array

    # Normalize pixel values to be in the range [0, 255]
    pixel_data = ((pixel_data - pixel_data.min()) / (pixel_data.max() - pixel_data.min()) * 255).astype(np.uint8)

    return pixel_data

# Streamlit app
st.title('Medical Image Analysis App')

# Radio buttons to select between image and DICOM file
file_type = st.radio("Select File Type", ("Image", "DICOM"))

if file_type == "Image":
    uploaded_file = st.file_uploader("Choose a medical image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = read_and_preprocess_image(uploaded_file)
        # st.image(image[0], caption='Uploaded Image', use_column_width=True)

        # Detect adversarial attacks
        is_adversarial = detect_adversarial(image)
        st.write(f"Is Adversarial? {is_adversarial}")

        if is_adversarial:
            # cleaned_image = remove_adversarial_attack(image, adversarial_image - image)

            # Classify the medical condition using the cleaned image
            medical_condition_prediction = classify_condition(image)

            # Explain the prediction
            explanation = explain_prediction(image, medical_condition_model)

            st.image(image, caption='Image', use_column_width=True)
            st.write('Medical Condition Prediction:', medical_condition_prediction)
            st.write('Explanation:', explanation)
        else:
            # Classify the medical condition using the original image
            medical_condition_prediction = classify_condition(image)

            # Explain the prediction
            explanation = explain_prediction(image, medical_condition_model)

            # st.image(image, caption='Original Image', use_column_width=True)
            st.write('Medical Condition Prediction:', medical_condition_prediction)

if file_type == "DICOM":
    uploaded_dico = st.file_uploader("Choose a DICOM file...", type=["dcm"])

    if uploaded_dico is not None:
        # Read and preprocess the DICOM file
        dicom_image = read_and_preprocess_dicom(io.BytesIO(uploaded_dico.read()))
        # Generate a random 256-bit key
        encryption_key = secrets.token_bytes(32)

        # Encrypt DICOM data
        encrypted_dicom_data = encrypt(dicom_image, encryption_key)

        # Convert encrypted data to base64 encoding for CSV
        encrypted_base64 = encrypted_dicom_data.hex()

        # Save encrypted data to CSV
        csv_data = {
            "DICOM_ID": 1,
            "Encrypted_Data": encrypted_base64
        }

        csv_filename = "encrypted_dicom_data.csv"
        with open(csv_filename, mode='w', newline='') as csvfile:
            fieldnames = ["DICOM_ID", "Encrypted_Data"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            writer.writerow(csv_data)

        st.write('Encrypted DICOM Data saved to:', csv_filename)
        
        # Decrypt DICOM data for further processing (if needed)
        decrypted_dicom_data = np.frombuffer(decrypt(encrypted_dicom_data, encryption_key), dtype=np.uint16)

        st.image(dicom_image, caption='Original DICOM Image', use_column_width=True)

        ds = pydicom.dcmread(uploaded_dico, force=True)

        st.write(ds)
       # st.write('Encrypted DICOM Data:', encrypted_dicom_data)
       # st.write('Decrypted DICOM Data:', decrypted_dicom_data)