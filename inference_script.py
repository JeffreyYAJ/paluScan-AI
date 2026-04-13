
import tensorflow as tf
from PIL import Image
import numpy as np

def run_keras_inference(image_path, model_path='malaria_model.h5'):
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    img = Image.open(image_path)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize((128, 128))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Perform inference
    prediction = model.predict(img_array)
    score = float(prediction[0][0])

    # Interpret the prediction with corrected logic: Assuming score is P(parasitized)
    class_names = {0: "Parasité", 1: "Sain"}
    if score < 0.5: # High score means parasitized
        label_id = 0 # Map to 'Parasité'
        label_name = class_names[label_id]
        confidence = score
    else: # Low score means uninfected (Sain)
        label_id = 1 # Map to 'Sain'
        label_name = class_names[label_id]
        confidence = 1 - score

    print(f"Keras Inference Result for {image_path}:")
    print(f"  Prediction: {label_name}")
    print(f"  Confidence: {confidence * 100:.2f}%")
    print(f"  Raw Score: {score:.4f}")

# Example usage:
run_keras_inference('/content/C100P61ThinF_IMG_20150918_144823_cell_160.png', 'malaria_model.h5')
