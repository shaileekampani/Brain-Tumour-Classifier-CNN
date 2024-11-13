import os
from flask import Flask, request, jsonify, render_template, url_for
import numpy as np
from PIL import Image
import tensorflow as tf
import random
import pickle
import openai
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder='mri-images')
app.config['UPLOAD_FOLDER'] = 'uploads'

# Hard-code the OpenAI API key


# Load CNN model (initialize as None)
CNN = None

def load_model():
    global CNN
    if CNN is not None:
        return
    try:
        # Load model structure
        model_json_path = 'models/CNN_structure.json'
        with open(model_json_path, 'r') as json_file:
            model_json = json_file.read()
        CNN = tf.keras.models.model_from_json(model_json)

        # Load and set model weights
        weights_path = 'models/CNN_weights.pkl'
        with open(weights_path, 'rb') as weights_file:
            CNN.set_weights(pickle.load(weights_file))

        # Compile the model
        CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    except Exception as e:
        logger.error(f"Error loading model: {e}")

def get_model_prediction(image_path):
    load_model()
    try:
        img = Image.open(image_path).resize((224, 224))
        if img.mode != 'RGB':
            img = img.convert('RGB')

        img_array = np.expand_dims(np.array(img), axis=0)
        prediction = CNN.predict(img_array)

        predicted_index = np.argmax(prediction[0])
        class_labels = ['glioma', 'meningioma', 'no tumor', 'pituitary']
        return class_labels[predicted_index]
    except Exception as e:
        logger.error(f"Error in get_model_prediction: {e}")
        return None

def get_ai_insights(predicted_label):
    try:
        prompt = f"The MRI scan analysis has identified the presence of a {predicted_label}. Please provide some insights or important information about this condition."
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        ai_message = response['choices'][0]['message']['content']
        return ai_message
    except Exception as e:
        logger.error(f"Error fetching AI insights: {e}")
        return "Sorry, I couldn't fetch insights at the moment."

# Route for the homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve a random image and its prediction
@app.route('/get-random-image', methods=['GET'])
def get_random_image():
    try:
        class_dirs = ['glioma', 'meningioma', 'notumor', 'pituitary']
        selected_class = random.choice(class_dirs)
        image_dir = os.path.join('mri-images', selected_class)
        image_name = random.choice(os.listdir(image_dir))
        image_path = os.path.join(image_dir, image_name)

        predicted_label = get_model_prediction(image_path)
        insights = get_ai_insights(predicted_label)
        
        web_accessible_image_path = url_for('static', filename=f'{selected_class}/{image_name}')
        return jsonify({
            'image_path': web_accessible_image_path,
            'predicted_label': predicted_label,
            'insights': insights
        })
    except Exception as e:
        logger.error(f"Error in get_random_image route: {e}")
        return jsonify({'error': 'An error occurred'}), 500

# Route for uploading an image and getting a prediction
@app.route('/upload-image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        try:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            predicted_label = get_model_prediction(file_path)
            insights = get_ai_insights(predicted_label)
            
            os.remove(file_path)
            return jsonify({'predicted_label': predicted_label, 'insights': insights})
        except Exception as e:
            logger.error(f"Error in upload_image route: {e}")
            return jsonify({'error': 'An error occurred'}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)
