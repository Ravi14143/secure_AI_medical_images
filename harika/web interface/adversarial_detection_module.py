import tensorflow as tf
import numpy as np

def detect_adversarial(image):

    adversarial_detection_model = tf.keras.models.load_model('ad_classification_model.h5')
    # Make predictions using your trained model
    predictions = adversarial_detection_model.predict(image)
    class_labels = ["Adversial_image","Non-Advarsial_image"]  

    # Get the predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label
