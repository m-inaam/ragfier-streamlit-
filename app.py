import streamlit as st
import numpy as np
import cv2
from PIL import ImageFont, Image, ImageDraw
import tensorflow as tf
import arabic_reshaper
from bidi.algorithm import get_display
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Load ARSL model
arsl_model = tf.keras.models.load_model('models/arsl_model.h5', compile=False)
arsl_categories = [
    ["ain", 'ع'], ["al", "ال"], ["aleff", 'أ'], ["bb", 'ب'], ["dal", 'د'], ["dha", 'ط'], ["dhad", "ض"], ["fa", "ف"],
    ["gaaf", 'جف'], ["ghain", 'غ'], ["ha", 'ه'], ["haa", 'ه'], ["jeem", 'ج'], ["kaaf", 'ك'], ["la", 'لا'],
    ["laam", 'ل'],
    ["meem", 'م'], ["nun", "ن"], ["ra", 'ر'], ["saad", 'ص'], ["seen", 'س'], ["sheen", "ش"], ["ta", 'ت'],
    ["taa", 'ط'],
    ["thaa", "ث"], ["thal", "ذ"], ["toot", ' ت'], ["waw", 'و'], ["ya", "ى"], ["yaa", "ي"], ["zay", 'ز']
]

# Load ASL model
asl_model = tf.keras.models.load_model("models/asl_Model.h5", compile=False)
asl_class_names = open("models/labels.txt", "r").readlines()

# Camera input for sign language
def get_camera_input():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture camera input.")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        st.image(img, channels="RGB", use_column_width=True)
        if st.button("Capture"):
            return frame
        if st.button("Close"):
            break
    cap.release()

# ASL prediction
def asl_predict(image):
    try:
        img_processed = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
        img_processed = np.asarray(img_processed, dtype=np.float32).reshape(1, 224, 224, 3)
        img_processed = (img_processed / 127.5) - 1
        prediction = asl_model.predict(img_processed)
        index = np.argmax(prediction)
        class_name = asl_class_names[index]
        confidence_score = prediction[0][index]
        return class_name[2:], confidence_score
    except Exception as e:
        return str(e)

# ARSL prediction
def arsl_predict(image):
    try:
        img_processed = cv2.resize(image, (64, 64))
        img_processed = np.array(img_processed, dtype=np.float32)
        img_processed = np.reshape(img_processed, (-1, 64, 64, 3))
        img_processed = img_processed.astype('float32') / 255.
        proba = arsl_model.predict(img_processed)[0]
        mx = np.argmax(proba)
        score = proba[mx] * 100
        res = arsl_categories[mx][0]
        sequence = arsl_categories[mx][1]
        reshaped_text = arabic_reshaper.reshape(sequence)
        bidi_text = get_display(reshaped_text)
        return res, score
    except Exception as e:
        return str(e)

st.title('Sign Language Recognition')

task = st.selectbox("Choose a task", ["ASL Prediction", "ARSL Prediction"])

if task == "ASL Prediction":
    st.write("Use the camera to capture ASL sign:")
    image = get_camera_input()
    if image is not None:
        result, confidence = asl_predict(image)
        st.write("Prediction:", result)
        st.write("Confidence:", confidence)

elif task == "ARSL Prediction":
    st.write("Use the camera to capture ARSL sign:")
    image = get_camera_input()
    if image is not None:
        result, score = arsl_predict(image)
        st.write("Prediction:", result)
        st.write("Score:", score)
