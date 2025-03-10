from fastapi import FastAPI, UploadFile, File, HTTPException
import requests
import os
from typing import List

app = FastAPI()

MODEL_URL = "http://localhost:8001/predict"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.post("/predict_signs/")
async def predict_signs(video: UploadFile = File(...)):
    """Receives a video, sends it to the model container, and returns the predicted words."""
    try:
        files = {"video": (video.filename, video.file, video.content_type)}
        response = requests.post(MODEL_URL, files=files)
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error from model container")
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate_sentence/")
async def generate_sentence(words: List[str]):
    """Takes a list of words and generates a meaningful sentence using Gemini API."""
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="Gemini API key not found")
    
    headers = {"Authorization": f"Bearer {GEMINI_API_KEY}"}
    prompt = f"Form a grammatically correct sentence using these words: {', '.join(words)}"
    
    try:
        response = requests.post(
            "https://generativelanguage.googleapis.com/v1/models/gemini-pro:generateText",
            json={"prompt": {"text": prompt}},
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=500, detail="Error from Gemini API")
        
        return response.json()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
