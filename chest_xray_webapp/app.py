from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask import send_from_directory

print("Starting Flask app...")


app = Flask(__name__)
model = load_model('model.h5')

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224),color_mode="grayscale") 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    preds = model.predict(x)[0]
    classes = ['COVID19', 'NORMAL', 'PNEUMONIA', 'TURBERCULOSIS']
    pred_class = classes[np.argmax(preds)]
    confidence = np.max(preds)

    return render_template(
    'result.html',
    label=pred_class,
    confidence=round(confidence * 100, 2),
    image_file=file.filename
)


if __name__ == '__main__':
    app.run(debug=True)
