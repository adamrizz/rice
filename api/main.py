import numpy as np
import tensorflow as tf
from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import io

# --- Konfigurasi dan Inisialisasi ---
app = FastAPI()

# Path ke model TFLite yang sudah ada di dalam folder api/
MODEL_PATH = "api/padi_model.tflite"

# Muat model TFLite dan alokasikan tensor
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    print("Model TFLite berhasil dimuat.")
except Exception as e:
    raise RuntimeError(f"Gagal memuat model TFLite: {e}")

# Dapatkan detail input dan output dari model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Konfigurasi kelas
CLASS_NAMES = ["Bacterial Leaf Blight", "Leaf Blast", "Leaf Scald", "Brown Spot", "Narrow Brown Spot", "Healthy"]

# --- Endpoints API ---
@app.get("/api")
async def root():
    return {"message": "API Klasifikasi Daun Padi TFLite siap digunakan."}

@app.post("/api/predict")
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File harus berupa gambar.")

    try:
        contents = await file.read()
        
        # Dapatkan ukuran input yang diharapkan oleh model
        _, height, width, _ = input_details[0]['shape']
        
        image = Image.open(io.BytesIO(contents)).convert("RGB").resize((width, height))
        
        # Ubah gambar menjadi array numpy dan normalisasi
        img_array = np.expand_dims(np.array(image, dtype=np.float32), axis=0) / 255.0

        # Atur tensor input, jalankan inferensi, dan dapatkan hasilnya
        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])
        
        # Proses output
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
        raise HTTPException(status_code=500, detail=f"Terjadi kesalahan saat memproses gambar: {str(e)}")