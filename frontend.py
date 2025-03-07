import streamlit as st
import cv2
import numpy as np
import asyncio
import websockets
import threading

# WebSocket Backend URL
BACKEND_URL = "ws://localhost:8000/ws"

# Function to capture video and send frames to backend
def video_stream():
    cap = cv2.VideoCapture(0)  # Open webcam
    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stframe = st.empty()  # Placeholder for video stream
    stop_signal = threading.Event()

    async def send_frame(frame):
        try:
            async with websockets.connect(BACKEND_URL) as ws:
                _, buffer = cv2.imencode(".jpg", frame)  # Encode frame
                await ws.send(buffer.tobytes())  # Send to backend
        except Exception as e:
            st.error(f"WebSocket Error: {e}")

    # Video capture loop
    while not stop_signal.is_set():
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Failed to capture frame.")
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert for display
        stframe.image(frame, channels="RGB")  # Display in Streamlit

        asyncio.run(send_frame(frame))  # Send frame to backend

    cap.release()  # Release webcam

# UI Elements in Streamlit
st.title("SignSpeak - AI-Powered Sign Language Translator")
st.markdown("### Show a sign gesture, and the system will recognize and translate it.")

# Start/Stop buttons
if st.button("Start Camera"):
    threading.Thread(target=video_stream, daemon=True).start()

if st.button("Stop Camera"):
    st.write("Stopping camera... (refresh the page if needed)")
