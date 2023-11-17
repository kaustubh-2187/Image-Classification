import streamlit as st
import tensorflow as tf
from keras.models import load_model
from PIL import Image

from util import classify

# Set the Title
st.title("Animal Classification")

# Set Header
st.header("Upload Image")

# Upload File
file = st.file_uploader('', type=['jpeg','jpg','png'])

# Load the Classifier for making predictions
model = load_model("animal_classifier.h5", compile=False)
model.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# load class names
with open("classnames.txt", "r") as f:
    class_names = [a.split('\n')[0] for a in f.readlines()]

# Display Image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # Classify Image
    class_name, conf_score = classify(image, model, class_names)

    # Write Classification
    st.write("## {}".format(class_name))
    st.write("## score : {}".format(conf_score))