import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from io import BytesIO
import json
import os

# ========== Page Configuration ==========
st.set_page_config(
    page_title="MNIST Digit Recognition",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== Custom CSS ==========
st.markdown("""
<style>
    /* Main theme colors - Baby Blue */
    :root {
        --primary: #4A90E2;
        --primary-light: #81B5F0;
        --primary-dark: #2A6DB8;
        --secondary: #C9E2FE;
        --background: #F0F8FF;
        --text: #333333;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #4A90E2, #81B5F0);
        padding: 2.5rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 25px rgba(74, 144, 226, 0.3);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.95;
    }
    
    /* Result card */
    .result-card {
        background: linear-gradient(135deg, #F0F8FF, #E6F2FF);
        padding: 2.5rem;
        border-radius: 20px;
        border: 3px solid #4A90E2;
        box-shadow: 0 10px 30px rgba(74, 144, 226, 0.2);
        text-align: center;
        margin: 2rem 0;
    }
    
    .prediction-digit {
        font-size: 6rem;
        font-weight: 800;
        color: #2A6DB8;
        margin: 1rem 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    .confidence-text {
        font-size: 1.5rem;
        color: #4A90E2;
        font-weight: 600;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.1);
        margin-bottom: 1rem;
        border-left: 4px solid #81B5F0;
    }
    
    .info-card h3 {
        color: #2A6DB8;
        margin-bottom: 0.8rem;
    }
    
    /* Feature map container */
    .feature-map-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(74, 144, 226, 0.1);
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #4A90E2, #81B5F0);
        color: white;
        border-radius: 50px;
        padding: 0.6rem 2.5rem;
        border: none;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(74, 144, 226, 0.4);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        margin-top: 3rem;
        border-top: 2px solid #E6F2FF;
        color: #666;
    }
    
    .footer a {
        color: #4A90E2;
        text-decoration: none;
        font-weight: 500;
    }
    
    .footer a:hover {
        color: #2A6DB8;
        text-decoration: underline;
    }
    
    /* Visitor counter */
    .visitor-counter {
        position: fixed;
        bottom: 20px;
        left: 20px;
        background: white;
        padding: 10px 20px;
        border-radius: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        font-size: 0.9rem;
        color: #333;
        z-index: 1000;
        border: 2px solid #C9E2FE;
    }
    
    /* Progress bar */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #4A90E2, #81B5F0);
    }
</style>
""", unsafe_allow_html=True)

# ========== Constants ==========
VISITOR_STATS_FILE = 'visitor_stats.json'

# ========== Helper Functions ==========
@st.cache_resource
def load_models():
    """Load the trained models with caching"""
    try:
        main_model = tf.keras.models.load_model('models/mnist_model_final.keras')
        feature_extractor = tf.keras.models.load_model('models/mnist_feature_extractor.keras')
        return main_model, feature_extractor
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

def preprocess_image(image):
    """Preprocess image for model prediction"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize to 28x28
    image = image.resize((28, 28))
    
    # Convert to array and normalize
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    return img_array, image

def get_feature_maps(feature_extractor, img_array):
    """Extract feature maps from convolutional layers"""
    try:
        # Create intermediate models
        conv1_model = tf.keras.models.Model(
            inputs=feature_extractor.input,
            outputs=feature_extractor.get_layer('conv1').output
        )
        
        conv2_model = tf.keras.models.Model(
            inputs=feature_extractor.input,
            outputs=feature_extractor.get_layer('conv2').output
        )
        
        # Get feature maps
        conv1_features = conv1_model.predict(img_array, verbose=0)
        conv2_features = conv2_model.predict(img_array, verbose=0)
        
        return conv1_features, conv2_features
    except Exception as e:
        st.error(f"Error extracting feature maps: {e}")
        return None, None

def create_feature_map_figure(features, layer_name, rows, cols):
    """Create a matplotlib figure for feature maps"""
    num_filters = features.shape[3]
    fig, axes = plt.subplots(rows, cols, figsize=(16, 8 if rows == 4 else 16))
    fig.suptitle(f'Feature Maps - {layer_name}', fontsize=16, fontweight='bold')
    
    for i in range(num_filters):
        row = i // cols
        col = i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.imshow(features[0, :, :, i], cmap='viridis')
        ax.axis('off')
    
    plt.tight_layout()
    return fig

def create_confidence_chart(predictions):
    """Create confidence bar chart for all digits"""
    digits = list(range(10))
    confidences = [predictions[0][i] * 100 for i in digits]
    
    # Color the predicted digit differently
    colors = ['#4A90E2' if c == max(confidences) else '#C9E2FE' for c in confidences]
    
    fig = go.Figure(data=[
        go.Bar(
            x=digits,
            y=confidences,
            marker=dict(color=colors, line=dict(color='white', width=2)),
            text=[f'{c:.1f}%' for c in confidences],
            textposition='auto',
            textfont=dict(size=12, color='white', family='Arial Black'),
        )
    ])
    
    fig.update_layout(
        title="Confidence Scores for All Digits",
        xaxis_title="Digit",
        yaxis_title="Confidence (%)",
        height=400,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,248,255,0.5)',
        font=dict(size=13, color='#333'),
        xaxis=dict(gridcolor='rgba(0,0,0,0.1)', dtick=1),
        yaxis=dict(gridcolor='rgba(0,0,0,0.1)')
    )
    
    return fig

def update_visitor_count():
    """Update and return visitor count"""
    if not os.path.exists(VISITOR_STATS_FILE):
        stats = {'total_visitors': 0}
    else:
        try:
            with open(VISITOR_STATS_FILE, 'r') as f:
                stats = json.load(f)
        except:
            stats = {'total_visitors': 0}
    
    if 'visited' not in st.session_state:
        stats['total_visitors'] = stats.get('total_visitors', 0) + 1
        st.session_state.visited = True
        
        try:
            with open(VISITOR_STATS_FILE, 'w') as f:
                json.dump(stats, f)
        except:
            pass
    
    return stats.get('total_visitors', 0)

# ========== Main Application ==========
def main():
    # Update visitor count
    visitor_count = update_visitor_count()
    
    # Visitor counter
    st.markdown(f"""
    <div class="visitor-counter">
        👥 <strong>{visitor_count}</strong> visitors
    </div>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔢 MNIST Digit Recognition</h1>
        <p>Upload a handwritten digit image and get real-time AI-powered predictions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/neural-network.png", width=100)
        st.title("Navigation")
        
        page = st.radio(
            "Select Page",
            ["🏠 Home - Recognize Digit", "📊 Model Information", "ℹ️ About"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### 📈 Model Stats")
        st.info("""
        **Architecture:** CNN
        **Accuracy:** ~99%
        **Dataset:** MNIST (70,000 images)
        **Input:** 28×28 grayscale
        **Classes:** 0-9 digits
        """)
        
        st.markdown("---")
        st.markdown("### 💡 Quick Tips")
        st.success("""
        - Use clear handwritten digits
        - Black digit on white background works best
        - Center the digit in the image
        - Avoid excessive noise
        """)
    
    # Main content based on selected page
    if page == "🏠 Home - Recognize Digit":
        show_recognition_page()
    elif page == "📊 Model Information":
        show_model_info_page()
    elif page == "ℹ️ About":
        show_about_page()
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p><strong>MNIST Digit Recognition System</strong> © 2024</p>
        <p>Developed by <a href="https://www.linkedin.com/in/mo-abdalkader/" target="_blank">Mohamed Abdalkader</a></p>
        <p>
            <a href="mailto:Mohameed.Abdalkadeer@gmail.com">📧 Email</a> | 
            <a href="https://github.com/Mo-Abdalkader" target="_blank">💻 GitHub</a>
        </p>
        <p style="font-size: 0.85rem; color: #999; margin-top: 1rem;">
            Built with TensorFlow, Keras, and Streamlit
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_recognition_page():
    """Main digit recognition page"""
    # Load models
    main_model, feature_extractor = load_models()
    
    if main_model is None or feature_extractor is None:
        st.error("❌ Models could not be loaded. Please check the model files.")
        return
    
    st.success("✅ Models loaded successfully!")
    
    # File uploader
    st.markdown("### 📤 Upload Handwritten Digit Image")
    uploaded_file = st.file_uploader(
        "Choose an image (PNG, JPG, JPEG)",
        type=['png', 'jpg', 'jpeg'],
        help="Upload a clear image of a handwritten digit (0-9)"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📸 Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True, caption="Original Image")
        
        with col2:
            st.markdown("#### 🔬 Analysis Options")
            
            show_feature_maps = st.checkbox("Show Feature Maps", value=False)
            show_all_predictions = st.checkbox("Show All Digit Confidences", value=True)
            show_preprocessed = st.checkbox("Show Preprocessed Image", value=True)
            
            analyze_button = st.button("🔍 Recognize Digit", type="primary", use_container_width=True)
        
        if analyze_button:
            with st.spinner("🔄 Analyzing digit..."):
                # Preprocess image
                processed_img, resized_img = preprocess_image(image)
                
                # Make prediction
                predictions = main_model.predict(processed_img, verbose=0)
                predicted_digit = np.argmax(predictions[0])
                confidence = float(predictions[0][predicted_digit]) * 100
                
                # Display results
                st.markdown("---")
                st.markdown("## 🎯 Recognition Results")
                
                # Result card
                st.markdown(f"""
                <div class="result-card">
                    <p style="font-size: 1.2rem; color: #666; margin-bottom: 0.5rem;">Predicted Digit</p>
                    <div class="prediction-digit">{predicted_digit}</div>
                    <p class="confidence-text">Confidence: {confidence:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Confidence bar
                st.markdown("#### Confidence Level")
                st.progress(confidence / 100)
                
                # Show preprocessed image
                if show_preprocessed:
                    st.markdown("---")
                    st.markdown("### 🖼️ Preprocessed Image (28×28)")
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.image(resized_img, width=300, caption="28×28 Grayscale")
                
                # All predictions chart
                if show_all_predictions:
                    st.markdown("---")
                    st.markdown("### 📊 Confidence for All Digits")
                    fig = create_confidence_chart(predictions)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Feature maps
                if show_feature_maps:
                    st.markdown("---")
                    st.markdown("### 🗺️ Neural Network Feature Maps")
                    st.info("Feature maps show the activations of convolutional filters at different layers")
                    
                    with st.spinner("Generating feature maps..."):
                        conv1_features, conv2_features = get_feature_maps(feature_extractor, processed_img)
                        
                        if conv1_features is not None and conv2_features is not None:
                            # Conv1 Feature Maps
                            st.markdown("#### Convolutional Layer 1 (32 filters)")
                            fig1 = create_feature_map_figure(conv1_features, "Conv Layer 1", 4, 8)
                            st.pyplot(fig1)
                            plt.close()
                            
                            # Conv2 Feature Maps
                            st.markdown("#### Convolutional Layer 2 (64 filters)")
                            fig2 = create_feature_map_figure(conv2_features, "Conv Layer 2", 8, 8)
                            st.pyplot(fig2)
                            plt.close()
                        else:
                            st.warning("Could not generate feature maps")

def show_model_info_page():
    """Model architecture and training information"""
    st.markdown("## 📊 Model Architecture & Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🏗️ Model Architecture</h3>
            <p><strong>Type:</strong> Convolutional Neural Network (CNN)</p>
            <p><strong>Input Shape:</strong> 28×28×1 (grayscale)</p>
            <p><strong>Output Classes:</strong> 10 (digits 0-9)</p>
            <br>
            <p><strong>Layer Structure:</strong></p>
            <ul>
                <li>Conv2D (32 filters, 3×3)</li>
                <li>MaxPooling2D (2×2)</li>
                <li>Conv2D (64 filters, 3×3)</li>
                <li>MaxPooling2D (2×2)</li>
                <li>Flatten</li>
                <li>Dense (64 units, ReLU)</li>
                <li>Dropout (0.5)</li>
                <li>Dense (10 units, Softmax)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📈 Training Details</h3>
            <p><strong>Dataset:</strong> MNIST (70,000 images)</p>
            <p><strong>Training Set:</strong> 60,000 images</p>
            <p><strong>Test Set:</strong> 10,000 images</p>
            <p><strong>Optimizer:</strong> Adam</p>
            <p><strong>Loss Function:</strong> Categorical Crossentropy</p>
            <p><strong>Batch Size:</strong> 128</p>
            <p><strong>Early Stopping:</strong> Yes (patience=5)</p>
            <p><strong>Learning Rate Reduction:</strong> Yes</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>🎯 Performance Metrics</h3>
            <p><strong>Test Accuracy:</strong> ~99%</p>
            <p><strong>Model Size:</strong> ~400KB</p>
            <p><strong>Inference Time:</strong> < 100ms</p>
            <p><strong>Total Parameters:</strong> ~150,000</p>
            <br>
            <p><strong>Key Features:</strong></p>
            <ul>
                <li>High accuracy on test data</li>
                <li>Fast inference speed</li>
                <li>Lightweight architecture</li>
                <li>Robust to variations</li>
                <li>Real-time predictions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🔬 Feature Extraction</h3>
            <p>The model uses two convolutional layers to extract features:</p>
            <br>
            <p><strong>Conv Layer 1 (32 filters):</strong></p>
            <p>Detects basic features like edges, lines, and curves</p>
            <br>
            <p><strong>Conv Layer 2 (64 filters):</strong></p>
            <p>Combines basic features into complex patterns that represent digit shapes</p>
            <br>
            <p>Each layer progressively learns more abstract representations of the input</p>
        </div>
        """, unsafe_allow_html=True)

def show_about_page():
    """About page with project information"""
    st.markdown("## ℹ️ About This Project")
    
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Project Overview</h3>
        <p>
            This MNIST Digit Recognition system uses deep learning and convolutional neural networks 
            to identify handwritten digits from images. The model is trained on the famous MNIST dataset 
            and achieves approximately 99% accuracy on test data.
        </p>
        <p>
            The project features both Flask and Streamlit web applications, providing users with 
            multiple ways to interact with the model and visualize its internal workings through 
            feature map analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="info-card">
            <h3>🤖 How It Works</h3>
            <p>
                <strong>1. Image Upload:</strong> User uploads a handwritten digit image<br><br>
                <strong>2. Preprocessing:</strong> Image is converted to 28×28 grayscale<br><br>
                <strong>3. Feature Extraction:</strong> CNN layers detect patterns<br><br>
                <strong>4. Classification:</strong> Fully connected layers predict the digit<br><br>
                <strong>5. Results:</strong> Display prediction with confidence score
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>🎓 Educational Value</h3>
            <p>This project demonstrates:</p>
            <ul>
                <li>Convolutional Neural Networks (CNNs)</li>
                <li>Image Classification with Deep Learning</li>
                <li>Feature Map Visualization</li>
                <li>Web Application Development</li>
                <li>TensorFlow/Keras Framework</li>
                <li>Model Deployment Strategies</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-card">
            <h3>💻 Technical Stack</h3>
            <ul>
                <li><strong>Deep Learning:</strong> TensorFlow/Keras</li>
                <li><strong>Web Framework (Option 1):</strong> Flask</li>
                <li><strong>Web Framework (Option 2):</strong> Streamlit</li>
                <li><strong>Visualization:</strong> Matplotlib, Plotly</li>
                <li><strong>Image Processing:</strong> Pillow (PIL)</li>
                <li><strong>Numerical Computing:</strong> NumPy</li>
                <li><strong>Language:</strong> Python 3.8+</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-card">
            <h3>📚 MNIST Dataset</h3>
            <p>
                The MNIST database (Modified National Institute of Standards and Technology) 
                is a large collection of handwritten digits commonly used for training and 
                testing in machine learning.
            </p>
            <p><strong>Dataset Statistics:</strong></p>
            <ul>
                <li>70,000 total images</li>
                <li>60,000 training samples</li>
                <li>10,000 test samples</li>
                <li>28×28 pixel grayscale images</li>
                <li>10 classes (digits 0-9)</li>
                <li>Balanced distribution</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>👨‍💻 Developer</h3>
        <p>
            Created by <strong>Mohamed Abdalkader</strong> - A passionate developer focused on 
            AI, machine learning, and creating intuitive user experiences.
        </p>
        <p>
            📧 Email: <a href="mailto:Mohameed.Abdalkadeer@gmail.com">Mohameed.Abdalkadeer@gmail.com</a><br>
            💼 LinkedIn: <a href="https://www.linkedin.com/in/mo-abdalkader/" target="_blank">mo-abdalkader</a><br>
            💻 GitHub: <a href="https://github.com/Mo-Abdalkader" target="_blank">Mo-Abdalkader</a>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-card">
        <h3>⚠️ Disclaimer</h3>
        <p>
            This application is for educational and demonstration purposes. While the model 
            achieves high accuracy on the MNIST test set, performance may vary with real-world 
            handwritten digits due to differences in writing styles, image quality, and preprocessing.
        </p>
    </div>
    """, unsafe_allow_html=True)

# Run the main application
main()
