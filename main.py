
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import time

app = Flask(__name__)
CORS(app)  

MODEL = tf.keras.models.load_model('malaria_model.h5')
CLASS_NAMES = {0: "Parasité", 1: "Sain"}

def prepare_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "Aucun fichier envoyé"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Nom de fichier vide"}), 400

    try:
        start_time = time.time()
        
        img_bytes = file.read()
        processed_img = prepare_image(img_bytes)
        
        prediction = MODEL.predict(processed_img)
        score = float(prediction[0][0])
        
        label_id = 1 if score > 0.5 else 0
        label_name = CLASS_NAMES[label_id]
        confidence = score if label_id == 1 else (1 - score)
        
        inference_time = (time.time() - start_time) * 1000

        return jsonify({
            "success": True,
            "prediction": label_name,
            "confidence": round(confidence * 100, 2),
            "inference_time_ms": round(inference_time, 2),
            "raw_score": round(score, 4)
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
