import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
from ultralytics import YOLO
model = YOLO("./best_1024_E6_150epoch.pt") 

st.title("My first Streamlit app")
st.write("Hello, world")


def callback(frame):
    img = frame.to_ndarray(format="bgr24")

    img = model(img,conf=0.4,iou= 0.5)

    # 結果をフレームに描画して表示
    img = img[0].plot()

    return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(key="example", video_frame_callback=callback)