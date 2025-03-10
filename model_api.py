import cv2
import numpy as np
import os
import torch
import tempfile
import requests
import uuid
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from elevenlabs import VoiceSettings
from elevenlabs.client import ElevenLabs

app = FastAPI()

model = torch.jit.load("sign_model.pth")
model.eval()

signs_to_words = {0: "hello", 1: "help", 2: "i", 3: "no", 4: "please", 
                  5: "sorry", 6: "stop", 7: "thank you", 8: "yes", 9: "you"}

GEMINI_API_KEY = ""
ELEVENLABS_API_KEY = ""

client = ElevenLabs(
  api_key='sk_6a32898ebb49e17e37094627ed85de9d903d6f3e4e5702bf',
)

def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)
    
    cap.release()
    return frames[:150]

def segment_video(frames):
    segments = []
    stride = 15
    for i in range(0, len(frames) - 16 + 1, stride):
        segments.append(np.array(frames[i:i+16]))
    return segments

def preprocess_frames(frames):
    frames = torch.tensor(frames).float()
    frames = frames.permute(0, 3, 1, 2)
    frames = frames / 255.0

    frames = frames.unsqueeze(0)
    frames = frames.permute(0, 2, 1, 3, 4)

    return frames

def predict_sign(segment):
    segment_tensor = preprocess_frames(segment)
    
    with torch.no_grad():
        output = model(segment_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
    
    return predicted_class

def generate_sentence(words):
    """Takes predicted words and generates a meaningful sentence using Gemini API."""
    if not GEMINI_API_KEY:
        return "Gemini API key not found"
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{"text": f"Form a grammatically correct 1-line meaningful sentence using these words: {', '.join(words)}"}]
        }]
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        
        if response.status_code != 200:
            return f"Error from Gemini API: {response.text}"
        
        return response.json().get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "Error: No response")
    
    except Exception as e:
        return str(e)

def text_to_speech_file(text: str) -> str:
    response = client.text_to_speech.convert(
        voice_id="pNInz6obpgDQGcFmaJgB",
        output_format="mp3_22050_32",
        text=text,
        model_id="eleven_turbo_v2_5",
        voice_settings=VoiceSettings(
            stability=0.0,
            similarity_boost=1.0,
            style=0.0,
            use_speaker_boost=True,
            speed=1.0,
        ),
    )
    save_file_path = f"{uuid.uuid4()}.mp3"
    with open(save_file_path, "wb") as f:
        for chunk in response:
            if chunk:
                f.write(chunk)

    print(f"{save_file_path}: A new audio file was saved successfully!")
    return save_file_path

@app.post("/predict")
async def predict(video: UploadFile = File(...)):
    """Processes video, predicts sign words, generates a sentence, and returns it with audio."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video.file.read())
        temp_video_path = temp_video.name
    
    try:
        frames = extract_frames(temp_video_path)
        segments = segment_video(frames)
        predictions = [predict_sign(segment) for segment in segments]
        seen = set()
        ordered_signs = [sign for sign in predictions if not (sign in seen or seen.add(sign))]
        ordered_signs_words = [signs_to_words[sign] for sign in ordered_signs]
        sentence = generate_sentence(ordered_signs_words)
        audio_file_path = text_to_speech_file(sentence)

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        os.remove(temp_video_path)
    
    return {
        "predicted_words": ordered_signs_words, 
        "generated_sentence": sentence,
        "audio_file": audio_file_path
    }

@app.get("/audio/{file_name}")
async def get_audio(file_name: str):
    """Endpoint to retrieve the generated audio file."""
    return FileResponse(file_name)