import streamlit as st
import torch
from detect import detect
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

confidence = st.slider("Select the mimimum confidence for image recongnition", min_value = 0, max_value = 1, step = .01)

@st.cache(persist=True)
def loadModel():
    start_time = time.time()
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
    finished_time = time.time()
    print(f"model downloaded, ETA:{finished_time-start_time})
