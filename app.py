import streamlit as st
from pathlib import Path
from PIL import Image

# Get relative path
img_path = Path(__file__).parents[0]
# Load images
logo_img = Image.open(f"{img_path}/Images/Logo.png")

# # Load a pretrained YOLO model (recommended for training)
# model = YOLO('yolov8n.pt')

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
    st.subheader("Select a sample video or upload your own!")
    sample_video = False
    # Create a column for each button
    button_l, button_r = st.columns((1, 1))

    # Add button one, which opens the first video when pressed
    with button_l:
        if st.button("Sample 1"):
            with open(f'{img_path}/Videos/Sample_1.mov', 'rb') as video_file:
                sample_video = video_file.read()

    # Add button two, which opens the second video when pressed
    with button_r:
        if st.button("Sample 2"):
            with open(f'{img_path}/Videos/Sample_2.mov', 'rb') as video_file:
                sample_video = video_file.read()

    # Add file uploader
    uploaded_video = st.file_uploader("Choose a file", type=["mp4", "mov"], accept_multiple_files=False)
    if uploaded_video:
        st.video(uploaded_video)
    if sample_video:
        st.video(sample_video)