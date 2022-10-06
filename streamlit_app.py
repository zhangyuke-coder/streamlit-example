import streamlit as st
import cv2
from PIL import Image, ImageEnhance
import numpy as np
import os
import trition
import tritonclient.grpc as grpclient
from image.process_image import image_detect
@st.cache
def load_image(img):
    im = Image.open(img)
    return im

def main():
    st.title("yolo detection App")
    st.text("Build with Streamlit and yolov7")
    
    activities = ["Detection", "About"]
    choice = st.sidebar.selectbox("Select Activty", activities)
    trition_client = trition.init_triton('127.0.0.1:8001')
    if choice == 'Detection':
        st.subheader("Yolo Detection")
        image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
        
        if image_file is not None:
            print("image_file:{}".format(type(image_file)))
            our_image = Image.open(image_file)
            print("our_image:{}".format(type(our_image)))
            st.text("Original Image")
            st.image(our_image)
            image_process = image_detect(trition_client, our_image)
            image_process.process()
    elif choice == 'About':
        st.subheader("About")
if __name__ == '__main__':
    main();