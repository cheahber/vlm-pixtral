import asyncio
import websockets
import numpy as np
import cv2
import streamlit as st
import requests


# Base URL of the FastAPI server
BASE_URL = "http://localhost:8000"

def start_processor():
    url = f"{BASE_URL}/start"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return "success", response.json().get("message", "Processor started successfully"), None
        else:
            return "error", response.json().get("detail", "Failed to start processor"), None
    except Exception as e:
        return "error", f"Failed to call start_processor: {e}", None

def stop_processor():
    url = f"{BASE_URL}/stop"
    try:
        response = requests.post(url)
        if response.status_code == 200:
            return "success", response.json().get("message", "Processor stopped successfully"), None
        else:
            return "error", response.json().get("detail", "Failed to stop processor"), None
    except Exception as e:
        return "error", f"Failed to call stop_processor: {e}", None

def update_video_path(video_path):
    url = f"{BASE_URL}/update-video-path"
    payload = {"video_path": video_path}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return "success", response.json().get("message", f"Video path updated to {video_path}"), None
        else:
            return "error", response.json().get("detail", "Failed to update video path"), None
    except Exception as e:
        return "error", f"Failed to call update_video_path: {e}", None

def update_prompt(prompt):
    url = f"{BASE_URL}/update-prompt"
    payload = {"prompt": prompt}
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return "success", response.json().get("message", f"Prompt updated to: {prompt}"), None
        else:
            return "error", response.json().get("detail", "Failed to update prompt"), None
    except Exception as e:
        return "error", f"Failed to call update_prompt: {e}", None


def display_system_status(video, model, prompt):
    st.markdown("### System Status")
    with st.container(border=True):
        st.write(f"**Uploaded Video:** {video if video else 'No video uploaded'}")
        st.write(f"**Selected Model:** {model}")
        st.write(f"**Prompt:** {prompt}")


# WebSocket URL of the FastAPI server
WEBSOCKET_URL = "ws://localhost:8000/ws/frames"


async def fetch_frames(placeholder, vlm_running, frame_display_interval):
    """
    Fetch frames from the WebSocket server and update the Streamlit placeholder.
    """
    try:
        async with websockets.connect(WEBSOCKET_URL) as websocket:
            while vlm_running:
                # Receive the frame bytes from the WebSocket
                frame_bytes = await websocket.recv()

                # Decode the JPEG bytes into an OpenCV image
                nparr = np.frombuffer(frame_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                # Convert BGR to RGB for Streamlit display
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Update the Streamlit placeholder
                placeholder.image(frame, channels="RGB", use_container_width=True)

                # Limit frame rate for display
                # await asyncio.sleep(frame_display_interval)
    except Exception as e:
        st.error(f"Error fetching frames: {e}")

# Streamlit Page Configuration
st.set_page_config(page_title="Video Wildfire Detection System", page_icon="🔥", layout="wide")

# Title Section
st.markdown("""
    <style>
        .title-container { text-align: center; margin-top: -30px; }
        h1 { font-size: 2.5rem; }
    </style>
    <div class="title-container"><h1>Video Wildfire Detection System</h1></div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.title("Settings")
model_name = st.sidebar.selectbox("Step 1: Choose a Model", ["Pixtral 12B"], key="model_dropdown")

start_vlm = st.sidebar.button("Start vlm")
stop_vlm = st.sidebar.button("Stop vlm")

# Real-time Frame Display with WebSocket
if start_vlm:
    status, message, data = start_processor()
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)
elif stop_vlm:
    status, message, data = stop_processor()
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# Video Upload
uploaded_file = st.sidebar.file_uploader("Step 2: Upload your video file", type=["mp4", "avi", "mov", "mkv"])

import tempfile
import os

video_path = None

if uploaded_file:
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, uploaded_file.name)
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getvalue())

    status, message, data = update_video_path(video_path)
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# Prompt Input
default_prompt = "Provide a concise JSON with key 'wildfire': True/False based on detection."
question = st.sidebar.text_area("Step 3: Customize the Prompt", default_prompt)
if st.sidebar.button("Submit Prompt"):
    status, message, data = update_prompt(question)
    if status == "success":
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

# Real-time Video Display Placeholder
with st.container(border=True, height=600):
    placeholder = st.empty()

frame_display_interval = 0.05  # Limit frame rate for smoother display

# Control Logic for VLM
st.sidebar.text("Step 4: Control VLM")
start_viewing = st.sidebar.button("Start viewing")
stop_viewing = st.sidebar.button("Stop viewing")

# Real-time Frame Display with WebSocket
if start_viewing:
    vlm_running = True
    st.info("Real-time video processing started. Press 'Stop viewing' to stop.")
    asyncio.run(fetch_frames(placeholder, vlm_running, frame_display_interval))
elif stop_viewing:
    vlm_running = False
    st.info("Real-time frame display stopped.")

# Display System Status
display_system_status(video_path, model_name, question)
