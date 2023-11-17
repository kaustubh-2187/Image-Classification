from PIL import ImageOps, Image
import numpy as np
import streamlit as st

def classify(image, model, class_names):

    image = image.resize((224,224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array,axis=0)

    data = np.ndarray(shape=(1,224,224,3), dtype=np.float32)
    data[0] = image_array

    # make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].capitalize()
    confidence_score = prediction[0][index]

    return class_name, confidence_score