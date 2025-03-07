import cv2

cap = cv2.VideoCapture(0)  # Try 1, 2, or -1 if 0 fails

if not cap.isOpened():
    print("Camera access failed. Try another index or check permissions.")
else:
    print("Camera is working!")

cap.release()
