import tensorflow as tf
import numpy as np

def classify_condition(image):
   
    medical_condition_model = tf.keras.models.load_model('classification_model.h5')
    # Make predictions using your trained model
    predictions = medical_condition_model.predict(image)

    class_labels = ["aneurysm","cancer","tumor"]  

    # Get the predicted class
    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]
    return predicted_class_label
