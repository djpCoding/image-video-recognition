import streamlit as st
import torch
#from yolov5detect import detect, annotation
#import detect2 as detect2
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
from torch import nn
from torchvision import transforms
#import pafy
#import youtube_dl
#import skimage
#import argparse
#import pytorchvideo
#from pytorchvideo.transforms.functional import (
#    uniform_temporal_subsample,
#    short_side_scale_with_boxes,
#    clip_boxes_to_image,
#)
#from torchvision.transforms._functional_video import normalize
#from pytorchvideo.data.ava import AvaLabeledVideoFramePaths
#from pytorchvideo.models.hub import slow_r50_detection
#from objectDetection import *

st.title('Object Detection')
st.markdown("This model runs using YOLOv5, one of the most popular vision AIs. The instance segmentation models are some of the fastest and most accurate in the world. The image processing available in this app is done in real time to demonstrate the speed and accuracy of the YOLOv5 vision AI.")
st.markdown("The following graph shows the performance and latency of YOLOv5 compared to other models. This is done with the Microsoft COCO large image dataset and pytorch latency.")
st.image('https://user-images.githubusercontent.com/61612323/204180385-84f3aca9-a5e9-43d8-a617-dda7ca12e54a.png')
st.subheader("This project takes an image from an upload or from a set of test images and creates boxes around the image. There are preset test videos that can also be explored.")
st.markdown("To begin exploring image detection in this project, follow the steps below and select options from the sidebar.")
st.subheader("Select the Options on the Sidebar")





def imageInput(source):
    confidence_bar = st.slider("Select the mimimum confidence for image recongnition", min_value = 0.00, max_value = 1.00, step = .01)
    if source == "User upload":
        image_file = st.file_uploader("Upload An Image", type =['png','jpeg','jpg','gif'])
        col1,col2 = st.columns(2)
        if image_file is not None:
            image = Image.open(image_file)
            submit = st.button("Run prediction")
            with col1:
                st.image(image, caption="Uploaded Image", use_column_width ='always')
            timestamp = datetime.timestamp(datetime.now())
            image_path = os.path.join('data/outputs',str(timestamp)+image_file.name)
            output_path = os.path.join('data/outputs', os.path.basename(image_path))
            with open(image_path, mode ='wb') as f:
                f.write(image_file.getbuffer())

            if submit:
                #Call the model
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                model.cpu()
                model.conf = confidence_bar
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
            if image_file is not None and submit:
                model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
                model.cpu()
                model.conf = confidence_bar
                prediction = model(image_file)
                prediction.render()
                for im in prediction.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption="Model Prediction")

def videoInputFinal(source):
    if source == 'From test set':
        video_path = glob.glob('data/videos/*')
        video_selection = st.selectbox('Select a video from the test set', ('Video 1 - Short', 'Video 2 - Long'))
        # st.slider('Select a video from the test set', min_value=1, max_value=len(video_path), step=1)
#        submit = st.button("Run prediction")
        if video_selection == 'Video 1 - Short':
            video_path = open('data/videos/short_sample.mp4', 'rb')
            video_bytes = video_path.read()
            st.video(video_bytes)
            st.write('Selected Video')
            submit = st.button("Run prediction")
            if submit:
                video_path = open('data/final/ShortFinal.mp4', 'rb')
                video_bytes = video_path.read()
                st.video(video_bytes)
                st.write('Predicted Video')
        elif video_selection == 'Video 2 - Long':
            video_path = open('data/videos/long_sample.mp4', 'rb')
            video_bytes = video_path.read()
            st.video(video_bytes)
            st.write('Selected Video')
            submit = st.button("Run prediction")
            if submit:
                video_path = open('data/final/LongFinal.mp4', 'rb')
                video_bytes = video_path.read()
                st.video(video_bytes)
                st.write('Predicted Video')
    else:
        st.write("File upload is not supported due to OpenCV MP4 video encoding. Video encodings via this method are not supported on HTML5 video players. Please reference GitHub to run your own videos on a local machine or through setting CUPA as your device when running cv2 and YOLOv5.")



st.cache(persist=True)
def build_model():
    net = cv2.dnn.readNet("config_files/yolov5s.onnx")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

net = build_model()

st.cache(persist=True)
def load_classes():
    class_list = []
    with open("config_files/classes.txt", "r") as f:
        class_list = [cname.strip() for cname in f.readlines()]
    return class_list

class_list = load_classes()

INPUT_WIDTH = 640
INPUT_HEIGHT = 640
colors = [(255, 255, 0), (0, 255, 0), (0, 255, 255), (255, 0, 0)]

def detection(image, net):
#    INPUT_WIDTH, INPUT_HEIGHT, _ = input_image.shape
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (INPUT_WIDTH, INPUT_HEIGHT), swapRB=True, crop=False)
    net.setInput(blob)
    preds = net.forward()
    return preds


def wrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

#    INPUT_WIDTH, INPUT_HEIGHT, _ = input_image.shape
    rows = output_data.shape[0]
    image_width, image_height, _ = input_image.shape
    x_factor = image_width / INPUT_WIDTH
    y_factor =  image_height / INPUT_HEIGHT

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= confidence_bar:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):

                confidences.append(confidence)

                class_ids.append(class_id)

                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])

    return result_class_ids, result_confidences, result_boxes

def format_yolov5(source):

    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source

    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)

    return result

def videoInput4(source):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov', 'avi','wmv','m4v'])
    if uploaded_video is not None:
        timestamp = datetime.timestamp(datetime.now())
        video_path = os.path.join('data/outputs',str(timestamp)+uploaded_video.name)
        output_path = os.path.join('data/outputs', os.path.basename(video_path))
        with open(video_path, mode='wb') as f:
            f.write(uploaded_video.read())
        st_video = open(video_path, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write('Uploaded Video')

        submit = st.button("Run prediction")
        if submit:
            capture = cv2.VideoCapture(video_path)
    #        st.write(capture)
    #        st.write(capture.read())
            start = time.time_ns()

            frame_count = 0
            total_frames = 0
            fps = -1
            img_array = []
            while True:
                _, frame = capture.read()
                if frame is None:
                    st.write("End of stream")
                    break

                frame = cv2.resize(frame,(640,640))
                inputImage = format_yolov5(frame)
    #            st.write(inputImage)
                outs = detection(frame, net)

                class_ids, confidences, boxes = wrap_detection(frame, outs[0])

                frame_count += 1
                total_frames += 1

                height, width, layers = frame.shape

                for (classid, confidence, box) in zip(class_ids, confidences, boxes):
                     color = colors[int(classid) % len(colors)]
                     cv2.rectangle(frame, box, color, 2)
    #                 st.image(frame)
                     cv2.rectangle(frame, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
                     cv2.putText(frame, class_list[classid], (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, .5, (0,0,0))

                if frame_count >= 30:
                    end = time.time_ns()
                    fps = 1000000000 * frame_count / (end - start)
                    frame_count = 0
                    start = time.time_ns()

                if fps > 0:
                    fps_label = "FPS: %.2f" % fps
                    cv2.putText(frame, fps_label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    #            st.image(frame)
#                cv2.imshow('frame',frame)
                img_array.append(frame)
#                st.image(frame)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter('data/outputs/video.mp4', fourcc, 1,(width, height)) #  fps = capture.get(cv2.CAP_PROP_FPS)
#            clip = ImageSequenceClip(img_array)
            for i in range(len(img_array)):
                video.write(img_array[i])
            video_path2 = open('data/outputs/video.mp4', 'rb')
            video_bytes2 = video_path2.read()
            st.video(video_bytes2)
            st.write('Uploaded Video')
#                cv2.imshow()
#                st.write(img_array[i])
#            video_path = os.path.join('data/outputs',str(timestamp)+video)
#            output_path = os.path.join('data/outputs', os.path.basename(video_path))
#            with open(video_path, mode='wb') as f:
#                f.write(video.read())
#            st_video = open(video_path, 'rb')
#            video_bytes = st_video.read()
#            st.video(video_bytes)
#            st.video(video)
#            result_video = video.release()
#            cv2.destroyAllWindwos()
#            st2_video = open(output_path, 'rb')
#            video_bytes2 = st2_video.read()
#            st.video(result_video)
#            st.video(video)
#            if cv2.waitKey(1) > -1:
#                st.write("finished by user")
#                break
#        predictions = net.forward()
#        output = predictions[0]
#        st.write(result_class_ids)
#        for i in range(len(result_class_ids)):
#
#            box = result_boxes[i]
#            class_id = result_class_ids[i]
#
#            color = colors[class_id % len(colors)]
#
#            conf  = result_confidences[i]
#
#            cv2.rectangle(image, box, color, 2)
#            cv2.rectangle(image, (box[0], box[1] - 20), (box[0] + box[2], box[1]), color, -1)
#            cv2.putText(image, class_list[class_id], (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

def videoInput(source):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov', 'avi','wmv','m4v'])
    if uploaded_video is not None:
        timestamp = datetime.timestamp(datetime.now())
        video_path = os.path.join('data/outputs',str(timestamp)+uploaded_video.name)
        output_path = os.path.join('data/outputs', os.path.basename(video_path))
#        vid = uploaded_video.name
        with open(video_path, mode='wb') as f:
            f.write(uploaded_video.read())
        st_video = open(video_path, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write('Uploaded Video')
        weights = os.path.join("weights",'yolov5s.pt')
        st.write(video_path)
        if st.button("Predict/Detect"):
#            with st.spinner(text = 'Please wait...'):
#                run(weights=weights, source=video_path, conf_thres = confidence, device='cpu')


#            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#            model.cpu
#            model.conf = confidence
#            prediction = model(video_path)
#            prediction.render()
            detect2(weights="models/yoloTrained.pt", source=video_path, device='cpu') #weights=model,
            st_video2 = open(outputpath, 'rb')
            video_bytes2 = st_video2.read()
            st.video(video_bytes2)
            st.write("Model Prediction")




def main():
    st.sidebar.title('Options')
    data_source = st.sidebar.radio("Select input source:", ['From test set', 'User upload'])
    option = st.sidebar.radio("Select input type:",['Image','Video'])

    if option == "Image":
        imageInput(data_source)

    elif option == "Video":
        videoInputFinal(data_source)

main()

@st.cache(persist=True)
def loadModel():
    start_time = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    finished_time = time.time()
    print(f"model downloaded, ETA:{finished_time-start_time}")
loadModel()
