# app.py
from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Load your trained model
model = load_model('models/fine_tuned_model.h5')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(save_path)
        
        # Preprocess and predict
        processed_img = preprocess_image(save_path)
        prediction = model.predict(processed_img)[0][0]
        confidence = max(prediction, 1 - prediction) * 100
        name = request.form.get('name', 'Guest')
        age = request.form.get('age', 'N/A')
        gender = request.form.get('gender', 'N/A')

        # result = {
        #     'filename': filename,
        #     'probability': float(prediction),
        #     'diagnosis': 'Non-Diabetic' if prediction > 0.5 else 'Diabetic',
        #     'confidence': confidence
        # }
        
        return render_template('result.html',filename = filename, prediction=float(prediction),confidence=confidence,name=name,age=age,gender=gender)
    
    return redirect(request.url)

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)