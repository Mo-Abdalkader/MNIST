import os
import shutil
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Create directories if they don't exist
os.makedirs('static/uploads', exist_ok=True)
os.makedirs('static/maps', exist_ok=True)

# Load the models
main_model = tf.keras.models.load_model('models/mnist_model_final.keras')
feature_extractor = tf.keras.models.load_model('models/mnist_feature_extractor.keras')


def preprocess_image(image_path):
    """Preprocess the image for model prediction"""
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28))

    img_300 = img.resize((300, 300))
    img_300.save('static/uploads/digit.png')

    img_array = np.array(img)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array


def clear_folder(folder_path):
    """Clear all files in the specified folder"""
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Error deleting {file_path}: {e}')


def save_feature_maps(image_array):
    """Extract and save feature maps from convolutional layers in specified layout"""
    # Create intermediate models to extract features from different layers
    conv1_layer_model = tf.keras.models.Model(
        inputs=feature_extractor.input,
        outputs=feature_extractor.get_layer('conv1').output
    )

    conv2_layer_model = tf.keras.models.Model(
        inputs=feature_extractor.input,
        outputs=feature_extractor.get_layer('conv2').output
    )

    # Get feature maps
    conv1_features = conv1_layer_model.predict(image_array)
    conv2_features = conv2_layer_model.predict(image_array)

    # Save Conv1 feature maps in a 4x8 grid layout
    plt.figure(figsize=(16, 8))
    num_filters1 = conv1_features.shape[3]  # Should be 32

    for i in range(num_filters1):
        plt.subplot(4, 8, i + 1)  # 4 rows, 8 columns
        plt.imshow(conv1_features[0, :, :, i], cmap='viridis')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('static/maps/FeaturesMapConv1.png')
    plt.close()

    # Save Conv2 feature maps in an 8x8 grid layout
    plt.figure(figsize=(16, 16))
    num_filters2 = conv2_features.shape[3]  # Should be 64

    for i in range(num_filters2):
        plt.subplot(8, 8, i + 1)  # 8 rows, 8 columns
        plt.imshow(conv2_features[0, :, :, i], cmap='viridis')
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('static/maps/FeaturesMapConv2.png')
    plt.close()


@app.route('/')
def index():
    # Clear previous uploads and feature maps
    clear_folder('static/uploads')
    clear_folder('static/maps')
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']

    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Save the original uploaded file
    upload_path = os.path.join('static/uploads', 'original_upload.png')
    file.save(upload_path)

    # Preprocess the image
    img_array = preprocess_image(upload_path)

    # Generate and save feature maps
    save_feature_maps(img_array)

    # Make prediction
    prediction = main_model.predict(img_array)
    predicted_class = np.argmax(prediction[0])
    confidence = float(prediction[0][predicted_class] * 100)

    return jsonify({
        'predicted_digit': int(predicted_class),
        'confidence': confidence,
        'image_path': 'static/uploads/digit.png'
    })


@app.route('/feature_maps')
def feature_maps():
    return render_template('feature_maps.html')


@app.route('/reset')
def reset():
    # Clear folders and return to home page
    clear_folder('static/uploads')
    clear_folder('static/maps')
    return jsonify({'success': True})


if __name__ == '__main__':
    app.run(debug=True)
