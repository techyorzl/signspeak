from fastapi import FastAPI, WebSocket
import numpy as np
import cv2
import asyncio

app = FastAPI()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            frame_data = await websocket.receive_bytes()  # Receive frame
            np_arr = np.frombuffer(frame_data, np.uint8)  # Convert to numpy array
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode to image
            # Process frame here (e.g., send to MediaPipe)
            cv2.imshow("Received Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        cv2.destroyAllWindows()
