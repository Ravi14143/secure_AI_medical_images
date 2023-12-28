import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic

# Load the trained model
model = tf.keras.models.load_model('classification_model.h5')

# Function to explain a prediction using LIME
def explain_prediction(image, model):
    explainer = lime_image.LimeImageExplainer()

    # Convert grayscale to RGB if needed
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # Lime expects the image to be in [0, 1] range
    normalized_image = image / 255.0

    # Function to predict with the model
    predict_fn = lambda x: model.predict(x)

    # Reshape the image to remove the extra dimension
    reshaped_image = np.squeeze(normalized_image)

    st.write(f"Shape of reshaped_image: {reshaped_image.shape}")

    # Explain the prediction using slic segmentation
    explanation = explainer.explain_instance(
        reshaped_image,
        predict_fn,
        top_labels=1,
        num_samples=1000,  
        segmentation_fn=lambda img: slic(img, n_segments=100, compactness=10.0)
    )

    if explanation is not None:
        # Get the segments and their weights
        segments = explanation.segments
        weights = explanation.local_exp[explanation.top_labels[0]]

        # Create a heatmap using the segments and weights
        heatmap = np.zeros_like(reshaped_image)
        for seg, weight in weights:
            heatmap[segments == seg] = weight

        # Clip values to [0.0, 1.0] range
        heatmap = np.clip(heatmap, 0.0, 1.0)

        # Overlay heatmap on the original image
        marked_image = mark_boundaries(reshaped_image, segments)
        heatmap_overlay = mark_boundaries(heatmap, segments)

        # Display the original image, heatmap, and overlay in Streamlit
        #st.image(reshaped_image, caption='Original Image', use_column_width=True)
        st.image(heatmap_overlay, caption='Heatmap Overlay', use_column_width=True)
        st.image(marked_image, caption='Original with Boundaries', use_column_width=True)
    else:
        st.write("LIME explanation is None. Check parameters and input.")

