from PIL import ImageOps, Image
import numpy as np
import tensorflow as tf
import streamlit as st

def classify(image, model, class_names):

    image = image.resize((224,224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = image_array

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    return class_name, confidence_score