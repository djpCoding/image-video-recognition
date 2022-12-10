import streamlit as st
import torch
#from detect import detect
from PIL import Image, ImageEnhance
from io import *
import glob
from datetime import datetime
import os
import wget
import time
import sys
import pandas as pd
import numpy as np
import cv2
from streamlit_embedcode import github_gist
import urllib
import urllib.request
import moviepy.editor as moviepy

st.title('Object Detection')
st.subheader("Select the Options on the Sidebar")

confidence = st.slider("Select the mimimum confidence for image recongnition", min_value = 0.00, max_value = 1.00, step = .01)


#    elif option == "Video":
#        videoInput(data_source)



def imageInput(source):
    if source == "User upload":
        image_file = st.file_uploader("Upload An Image", type =['png','jpeg','jpg','gif'])
        col1,col2 = st.columns(2)
        if image_file is not None:
            image = Image.open(image_file)
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width ='always')
            timestamp = datetime.timestamp(datetime.now())
            image_path = os.path.join('data/outputs',str(timestamp)+image_file.name)
            output_path = os.path.join('data/outputs', os.path.basename(image_path))
            with open(image_path, mode ='wb') as f:
                f.write(image_file.getbuffer())

            #Call the model
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            model.cpu()
            model.conf = confidence
            prediction = model(image_path)
            prediction.render()
            for im in prediction.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(output_path)
            img_ = Image.open(output_path)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')


        elif source == 'From test set':
            image_path = glob.glob('data/images/*')
            image_selection = st.slider('Select a random image from the test set', min_value=1, max_value=len(image_path), step=1)
            image_file = image_path[image_selection-1]
            submit = st.button("Run prediction")
            col1, col2 = st.columns(2)
            with col1:
                image = Image.open(image_file)
                st.image(image, caption="Selected image", use_column_width='always')
            with col2:
                if image_fle is not None and submit:
                    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                    model.cpu()
                    model.conf = confidence
                    prediction = model(image_file)
                    prediction.render()
                    for im in prediction.ims:
                        im_base64 = Image.fromarray(im)
                        im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                        img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                        st.image(img_, caption="Model Prediction")


def main():
    st.sidebar.title('Options')
    data_source = st.sidebar.radio("Select input source:", ['From test set', 'User upload'])
    option = st.sidebar.radio("Select input type:",['Image','Video'])

    if option == "Image":
        imageInput(data_source)

main()

@st.cache(persist=True)
def loadModel():
    start_time = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    finished_time = time.time()
    print(f"model downloaded, ETA:{finished_time-start_time}")
loadModel()
