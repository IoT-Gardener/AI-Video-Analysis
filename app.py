import cv2
import numpy as np
import streamlit as st
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# Get relative path
img_path = Path(__file__).parents[0]
# Load images
logo_img = Image.open(f"{img_path}/Images/Logo.png")

# Set the page title and icon and set layout to "wide" to minimise margains
st.set_page_config(page_title="AI-Video-Analysis", page_icon=":camera_with_flash:")

with st.container():
    # Create columns
    head_l, head_r = st.columns((2.5, 1))

    with head_l:
        # Add a subheader
        st.subheader("Advancing Analytics")
        # Add a title
        st.title("AI Video Analysis Tool")

    with head_r:
        # Add logo
        st.image(logo_img)

    # Add description
    st.write("Understanding what is happening in video footage is key to reducing problems, accidents, and monitoring environments remotely!")
    st.write("This tool will allow you to upload short sections of video that will be annotated and described using Generative AI!")
    # Add spacer
    st.write("---")

with st.container():
    # Select the file to use
    option = st.selectbox('Please select the sample you wish to analyse', ('Sample_1', 'Sample_2'))

    if st.button("Run!"):
        with st.status("Loading resources") as status:
            st.write("Loading model: YOLOv8n...")
            model = YOLO("yolov8n.pt")

            # Open the video file
            st.write("Opening video...")
            video_path = f"{img_path}/Videos/{option}.mov"

        # Open the video
        vid_cap = cv2.VideoCapture(video_path)
        # Get the number of frames
        no_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create empty streamlit frame to display results
        st_frame = st.empty()

        # Create a progress bar
        progress_text = "Processing frames..."
        prog_bar = st.progress(0, text=progress_text)
        # Calculate the progress each frames makes
        percent_prog = 1 / no_frames
        # Create frame counter
        frame_ctr = 1

        # Loop through the video frames
        while vid_cap.isOpened():
            success, image = vid_cap.read()
            if success:
                res = model.predict(image, conf=0.4)
                result_tensor = res[0].boxes
                res_plotted = res[0].plot()
                st_frame.image(res_plotted,
                               caption='Detected Video',
                               channels="BGR",
                               use_column_width=True
                               )
                prog_bar.progress(frame_ctr*percent_prog, text=progress_text)
                frame_ctr += 1
            else:
                vid_cap.release()
                break