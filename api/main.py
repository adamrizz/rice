import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Tetap penting untuk tensorflow-cpu

import requests
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io
import numpy as np

# --- Konfigurasi dan Inisialisasi ---
app = FastAPI()

# URL dan path model
MODEL_URL = "https://github.com/adamrizz/padi-cnn/releases/download/v1.0/daun_padi_cnn_model.keras"
MODEL_PATH_LOCAL = "/tmp/daun_padi_cnn_model.keras" # Gunakan folder /tmp di Vercel

# Fungsi download model dari GitHub
def download_model(url, destination):
    if not os.path.exists(destination):
        print(f"Mengunduh model dari {url}...")
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(destination, "wb") as f:
                f.write(response.content)
            print("Model berhasil diunduh.")
        else:
            raise RuntimeError("Gagal mengunduh model.")

# Unduh dan load model
download_model(MODEL_URL, MODEL_PATH_LOCAL)
model = tf.keras.models.load_model(MODEL_PATH_LOCAL)
print("Model berhasil dimuat.")

# Konfigurasi kelas
IMAGE_HEIGHT, IMAGE_WIDTH = 150, 150
CLASS_NAMES = ["Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald", "Brown Spot", "Narrow Brown Spot", "Healthy"]

# --- Endpoints API ---
@app.get("/api")
async def root():
    return {"message": "API Klasifikasi Daun Padi siap digunakan."}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")
    
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((IMAGE_WIDTH, IMAGE_HEIGHT))
        img_array = np.expand_dims(np.array(image), axis=0) / 255.0

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        
        predicted_index = int(np.argmax(score))
        predicted_label = CLASS_NAMES[predicted_index]
        confidence = float(np.max(score))

        return {
            "filename": file.filename,
            "predicted_class": predicted_label,
            "confidence": confidence,
            "all_predictions": {CLASS_NAMES[i]: float(score[i]) for i in range(len(CLASS_NAMES))}
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan: {str(e)}")