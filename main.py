from fastapi import FastAPI, File, UploadFile, Depends, HTTPException, status
from typing import Optional
from jose import JWTError, jwt
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
from datetime import datetime, timedelta
from preprocess import preprocess_image
from gradcam_codes import generate_gradcam
from config import class_names 


# FastAPI App Initialization
app = FastAPI(
    title="Pneumonia Detection API",
    version="1.0.0",
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Make 'uploads' folder 
os.makedirs("uploads", exist_ok=True)

# Routes
@app.get("/")
def read_root():
    #Check if the FastAPI backend is running.
    return {"message": "FastAPI Backend is working"}


@app.post("/predict") 

async def predict_image_and_gradcam(file: UploadFile = File(...)):
    # Receives an image file, preprocesses it, performs prediction, and generates Grad-CAM visualization.
    # Save uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    try:
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Perform preprocessing and prediction
        img_array, pil_image = preprocess_image(file_path)
        response = generate_gradcam(file_path, img_array) 
        return JSONResponse(response)
    
    except Exception as e:
        # Log the full traceback on the server for debugging
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Prediction failed on server: {e}")
    finally:
        # Clean up the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
