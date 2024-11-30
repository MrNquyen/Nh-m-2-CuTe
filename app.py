# Ultralytics YOLO 🚀, AGPL-3.0 license

import streamlit as st

import cv2
import torch
import requests
import os
from PIL import Image
import numpy as np 

from ultralytics.utils.checks import check_requirements

def download_file(file_id, destination):
    url = f"https://drive.google.com/uc?id={file_id}"
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        directory = os.path.dirname(destination)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with open(destination, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    else:
        print(f"Failed to download file with ID {file_id}. Status code: {response.status_code}")
        return False
file_ids = ["1OsAa-MryW-4lNvzweL2xxw2qjn4yyUk4",
            '1aOEL9i87bleMUyfQPHnb-f1Awyfc0q_K',
            '1BgyfGvFiqOpIWqHDH-FPnlbA5L-1TR7B',
            '1HZlxtAUO1utwc2p9Rl5g5a54GJpNsdiS']
destination_paths = ["best.pt", 'logo.png', 'modelcomparison.jpg','authorproject.jpg']
for file_id, destination in zip(file_ids, destination_paths):
    if download_file(file_id, destination):
        st.success(f"Downloaded {destination} successfully.")
    else:
        st.error(f"Failed to download {destination}.")
    
def inference(model=None):
    """Performs object detection on your image using YOLO11n-cls."""
    check_requirements("streamlit>=1.29.0")  
    import streamlit as st

    from ultralytics import YOLO

    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    main_title_cfg = """<div><h1 style="color:#00CCFF; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Ultralytics YOLO Streamlit Application
                    </h1></div>"""

    sub_title_cfg = """<div><h4 style="color:#40e0d0; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Classify your image with the power of 2-day-non-sleep 🥲</h4>
                    </div>"""


    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    with st.sidebar:
        logo = "logo.png"
        st.image(logo, width=250)

    sidebar_option = st.sidebar.selectbox("HOME", ['🍀 About Us','🤖 User Configuration'])

    if sidebar_option == "🤖 User Configuration":
        uploaded_image = st.sidebar.file_uploader('Select Image', type=['jpg','jpeg','png'])
        model = YOLO('best.pt')
        st.success('Model loaded successfully!')
        conf = float(st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.3, 0.01))
        if st.sidebar.button("Classify Image") and uploaded_image is not None:
            image = Image.open(uploaded_image)
            image =cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            results = model(image, conf=conf)
            if results:
                st.success(results)
            else:
                st.warning("Object(s) in given image is undetectable.")
        torch.cuda.empty_cache()
    
    elif sidebar_option == '🍀 About Us':
        st.title('About Us')
        st.header('Introduction')
        st.write("""
                  ***The main goal of this study is to build a model capable of automatic 
                  waste sorting based on images, which helps to optimize the waste sorting 
                  and treatment process. This model not only contributes to reducing manual 
                  labor but also improves efficiency in waste management, towards a green, 
                  clean, smart and sustainable earth. With following specific goals:***
                  """)
        st.header('How does it works?')
        st.subheader('Transfer Learning')
        st.write("""
                  Transfer Learning is a machine learning technique in which a model 
                  developed for a specific task is reused as a starting point for a model 
                  on a second related task. This approach is especially useful when the 
                  second task has limited data, allowing the model to leverage the knowledge 
                  gained from the first task to improve performance for the new task.
                  """)
        st.subheader('Computer Vision')
        st.write("""
                  Computer vision is a rapidly growing field that gives machines the ability 
                  to interpret and understand image data. An important aspect in this field 
                  is object detection, which is the accurate identification and positioning 
                  of objects in an image or video. In recent years, algorithms have made 
                  significant strides in addressing this challenge.
                  """)
        st.subheader('YOLO')
        st.write("""
                  An important breakthrough in object detection was the introduction of the 
                  YOLO (You Only Look Once) platform introduced by Redmon et al. in 2015. 
                  This innovative method, as the name implies, processes the entire image in a 
                  single go to detect objects and their location. YOLO's approach differs from 
                  traditional two-stage detection methods by treating object detection as a 
                  regression problem. It uses a single convolutional neural network to 
                  simultaneously predict bounding boxes and classification probabilities across 
                  the entire image, simplifying the detection process compared to more complex 
                  methods in the past.
                  """)
        st.subheader('YOLO version 11')
        st.write("""
                  YOLOv11 is the latest version in the YOLO series, built on the foundation of 
                  YOLOv1. Announced at the YOLO Vision 2024 (YV24) conference, YOLOv11 marks an 
                  important step forward in real-time object detection technology. This new 
                  version brings major improvements in both architecture and training methods, 
                  promoting accuracy, speed, and efficiency.
                  """)
        st.subheader('Why did we choose YOLO11n-cls instead of other classic Transfer models?')
        st.image('modelcomparison.jpg', caption = 'Comparison between YOLO11n-cls and EfficientNet_B7')
        st.write("""
                   YOLO11n-cls is completely superior to EfficientNet_B7 in all indicators (Precision, 
                   Recall, F1-score). This shows that this model is much more suitable for the task of 
                   sorting waste types. EfficientNet_B7 does not meet the requirements with very low 
                   performance, which may be due to not being optimized or not suitable for this 
                   dataset.
                  """)
        st.header('Authors')
        st.image('authorproject.jpg')

if __name__ == "__main__":
    inference()