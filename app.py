# Ultralytics YOLO 🚀, AGPL-3.0 license

import streamlit as st

import cv2
import torch
import requests
import os
from PIL import Image
import numpy as np 

from ultralytics.utils.checks import check_requirements
from ultralytics.utils.downloads import GITHUB_ASSETS_STEMS

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
            '1aOEL9i87bleMUyfQPHnb-f1Awyfc0q_K']
destination_paths = ["best.pt", 'logo.png']
for file_id, destination in zip(file_ids, destination_paths):
    if download_file(file_id, destination):
        st.success(f"Downloaded {destination} successfully.")
    else:
        st.error(f"Failed to download {destination}.")
    
def inference(model=None):
    """Performs object detection on your image using YOLO11n-cls."""
    check_requirements("streamlit>=1.29.0")  # scope imports for faster ultralytics package load speeds
    import streamlit as st

    from ultralytics import YOLO

    # Hide main menu style
    menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

    # Main title of streamlit application
    main_title_cfg = """<div><h1 style="color:#00CCFF; text-align:center; font-size:40px; 
                             font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                    Ultralytics YOLO Streamlit Application
                    </h1></div>"""

    # Subtitle of streamlit application
    sub_title_cfg = """<div><h4 style="color:#40e0d0; text-align:center; 
                    font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
                    Classify your image with the power of 2-day-non-sleep 🥲</h4>
                    </div>"""

    # Set html page configuration
    st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

    # Append the custom HTML
    st.markdown(menu_style_cfg, unsafe_allow_html=True)
    st.markdown(main_title_cfg, unsafe_allow_html=True)
    st.markdown(sub_title_cfg, unsafe_allow_html=True)

    # Add logo in sidebar
    with st.sidebar:
        logo = "logo.png"
        st.image(logo, width=250)

    # Add elements to vertical setting menu
    st.sidebar.title("User Configuration")

    uploaded_image = st.sidebar.file_uploader('Select Image', type=['jpg','jpeg','png'])

    model = YOLO('best.pt')
    st.success('Model loaded successfully!')

    conf = float(st.sidebar.slider('Confidence Threshold', 0.0, 1.0, 0.3, 0.01))

    if st.sidebar.button("Classify Image") and uploaded_image is not None:
        image = Image.open(uploaded_image)
        image =cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BRG)

        results = model(image, conf=conf)
        class_name = list(model.name.values())

        if results:
            #first_result = results[0]
            #class_name = first_result.names[first_result.boxes[0].item()]
            #confidence_score = first_result.boxes[0].conf[0].item()

            #eco_label = 'biodegradable' if class_name in ['paper_waste', 'leaf_waste', 'food_waste', 'wood_waste'] else 'non-biodegradable'

            # Display the prediction result
            #st.success(f'Given image is {class_name} ({confidence_score:.2f}), {eco_label}.')
            st.success(results)
        else:
            st.warning("Object(s) in given image is undetectable.")


    # Clear CUDA memory
    torch.cuda.empty_cache()


# Main function call
if __name__ == "__main__":
    inference('best.pt')