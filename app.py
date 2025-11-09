from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__, static_folder='.', static_url_path='')

MODEL_PATH = 'pneumonia_model.h5'
try:
    model = load_model(MODEL_PATH)
    print("✓ Model loaded successfully!")
except Exception as e:
    print(f"✗ Error loading model: {e}")
    model = None

IMG_SIZE = 64

@app.route('/')
def index():
    html = '''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Pneumonia Detector AI</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            body {
                font-family: 'Arial', sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            .container {
                background: white;
                border-radius: 20px;
                padding: 50px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
                max-width: 600px;
                width: 100%;
                text-align: center;
            }
            h1 {
                color: #0066cc;
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .subtitle {
                color: #666;
                font-size: 1.1em;
                margin-bottom: 40px;
            }
            .upload-section {
                border: 3px dashed #0066cc;
                border-radius: 15px;
                padding: 40px;
                background: #f0f7ff;
                margin-bottom: 30px;
                cursor: pointer;
                transition: all 0.3s ease;
            }
            .upload-section:hover {
                background: #e6f2ff;
                border-color: #0052a3;
            }
            .upload-icon {
                font-size: 3em;
                margin-bottom: 15px;
            }
            #fileInput {
                display: none;
            }
            .upload-text {
                color: #333;
                font-size: 1.1em;
                font-weight: 600;
            }
            .upload-subtext {
                color: #999;
                font-size: 0.9em;
                margin-top: 10px;
            }
            .preview {
                margin: 30px 0;
                display: none;
            }
            .preview.active {
                display: block;
            }
            .preview img {
                max-width: 100%;
                max-height: 300px;
                border-radius: 10px;
                border: 2px solid #e0e0e0;
                padding: 10px;
            }
            button {
                width: 100%;
                padding: 15px;
                font-size: 1.1em;
                border: none;
                border-radius: 10px;
                cursor: pointer;
                font-weight: 600;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 15px;
            }
            .btn-analyze {
                background: #28a745;
                color: white;
            }
            .btn-analyze:hover:not(:disabled) {
                background: #218838;
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(40, 167, 69, 0.4);
            }
            .btn-analyze:disabled {
                background: #ccc;
                cursor: not-allowed;
                opacity: 0.6;
            }
            .btn-another {
                background: #17a2b8;
                color: white;
                display: none;
            }
            .btn-another:hover {
                background: #138496;
                transform: translateY(-2px);
            }
            .loading {
                display: none;
                text-align: center;
                margin: 30px 0;
            }
            .spinner {
                border: 4px solid #f3f3f3;
                border-top: 4px solid #0066cc;
                border-radius: 50%;
                width: 50px;
                height: 50px;
                animation: spin 1s linear infinite;
                margin: 0 auto 15px;
            }
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
            .result {
                display: none;
                margin: 30px 0;
                padding: 30px;
                border-radius: 15px;
                text-align: center;
            }
            .result.active {
                display: block;
            }
            .result.pneumonia {
                background: linear-gradient(135deg, #ffe0e0 0%, #ffcccc 100%);
                border: 2px solid #dc3545;
            }
            .result.normal {
                background: linear-gradient(135deg, #e0ffe0 0%, #ccffcc 100%);
                border: 2px solid #28a745;
            }
            .result-icon {
                font-size: 3em;
                margin-bottom: 10px;
            }
            .result-text {
                font-size: 1.8em;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .result.pneumonia .result-text {
                color: #dc3545;
            }
            .result.normal .result-text {
                color: #28a745;
            }
            .confidence-bar {
                margin: 20px 0;
            }
            .confidence-label {
                font-weight: 600;
                margin-bottom: 8px;
            }
            .bar {
                width: 100%;
                height: 30px;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
            }
            .bar-fill {
                height: 100%;
                background: linear-gradient(90deg, #0066cc, #00d4ff);
                transition: width 0.5s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
            }
            .result.pneumonia .bar-fill {
                background: linear-gradient(90deg, #dc3545, #ff6b6b);
            }
            .result.normal .bar-fill {
                background: linear-gradient(90deg, #28a745, #51cf66);
            }
            .recommendation {
                background: white;
                padding: 15px;
                border-radius: 10px;
                margin-top: 15px;
                font-weight: 600;
                font-size: 1em;
            }
            .error {
                background: #f8d7da;
                color: #721c24;
                padding: 15px;
                border-radius: 10px;
                margin: 15px 0;
                display: none;
            }
            .error.active {
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🫁 Pneumonia Detector AI</h1>
            <p class="subtitle">Upload a chest X-ray image for instant AI analysis</p>
            
            <div class="error" id="error"></div>
            
            <div class="upload-section" onclick="document.getElementById('fileInput').click()">
                <div class="upload-icon">📤</div>
                <div class="upload-text">Click to Upload or Drag & Drop</div>
                <div class="upload-subtext">JPG, PNG, GIF - Max 10MB</div>
            </div>
            
            <input type="file" id="fileInput" accept="image/*">
            
            <div class="preview" id="preview">
                <img id="previewImg" src="">
            </div>
            
            <button class="btn-analyze" id="analyzeBtn" onclick="analyzeImage()" disabled>
                🔍 Analyze X-ray
            </button>
            
            <button class="btn-another" id="anotherBtn" onclick="reset()">
                📤 Analyze Another
            </button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <div style="color: #666; font-weight: 600;">Analyzing X-ray... Please wait</div>
            </div>
            
            <div class="result" id="result">
                <div class="result-icon" id="resultIcon"></div>
                <div class="result-text" id="resultText"></div>
                <div class="confidence-bar">
                    <div class="confidence-label">Confidence Level</div>
                    <div class="bar">
                        <div class="bar-fill" id="barFill" style="width: 0%"></div>
                    </div>
                </div>
                <div class="recommendation" id="recommendation"></div>
            </div>
        </div>

        <script>
            const fileInput = document.getElementById('fileInput');
            const preview = document.getElementById('preview');
            const previewImg = document.getElementById('previewImg');
            const analyzeBtn = document.getElementById('analyzeBtn');
            const loading = document.getElementById('loading');
            const result = document.getElementById('result');
            const errorDiv = document.getElementById('error');
            
            // Handle drag and drop
            document.addEventListener('dragover', (e) => {
                e.preventDefault();
                document.querySelector('.upload-section').style.borderColor = '#28a745';
            });
            
            document.addEventListener('dragleave', () => {
                document.querySelector('.upload-section').style.borderColor = '#0066cc';
            });
            
            document.addEventListener('drop', (e) => {
                e.preventDefault();
                document.querySelector('.upload-section').style.borderColor = '#0066cc';
                handleFile(e.dataTransfer.files[0]);
            });
            
            fileInput.addEventListener('change', (e) => {
                handleFile(e.target.files[0]);
            });
            
            function handleFile(file) {
                if (!file || !file.type.startsWith('image/')) {
                    showError('Please select a valid image file');
                    return;
                }
                
                if (file.size > 10 * 1024 * 1024) {
                    showError('File size must be less than 10MB');
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = (e) => {
                    previewImg.src = e.target.result;
                    preview.classList.add('active');
                    analyzeBtn.disabled = false;
                    result.classList.remove('active');
                    hideError();
                };
                reader.readAsDataURL(file);
            }
            
            async function analyzeImage() {
                if (!fileInput.files[0]) {
                    showError('No file selected');
                    return;
                }
                
                analyzeBtn.disabled = true;
                loading.style.display = 'block';
                result.classList.remove('active');
                hideError();
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                try {
                    const response = await fetch('/api/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    loading.style.display = 'none';
                    
                    if (response.ok) {
                        displayResult(data);
                        result.classList.add('active');
                        document.getElementById('anotherBtn').style.display = 'block';
                    } else {
                        showError('Error: ' + (data.error || 'Analysis failed'));
                    }
                } catch (error) {
                    loading.style.display = 'none';
                    showError('Connection error: ' + error.message);
                    console.error(error);
                }
                
                analyzeBtn.disabled = false;
            }
            
            function displayResult(data) {
                document.getElementById('resultIcon').textContent = data.icon;
                document.getElementById('resultText').textContent = data.diagnosis;
                document.getElementById('recommendation').textContent = data.recommendation;
                
                result.className = 'result active ' + data.type;
                
                const barFill = document.getElementById('barFill');
                barFill.style.width = '0%';
                setTimeout(() => {
                    barFill.style.width = data.confidence + '%';
                    barFill.textContent = data.confidence + '%';
                }, 100);
            }
            
            function reset() {
                fileInput.value = '';
                preview.classList.remove('active');
                result.classList.remove('active');
                document.getElementById('anotherBtn').style.display = 'none';
                analyzeBtn.disabled = true;
                hideError();
            }
            
            function showError(msg) {
                errorDiv.textContent = msg;
                errorDiv.classList.add('active');
            }
            
            function hideError() {
                errorDiv.classList.remove('active');
            }
        </script>
    </body>
    </html>
    '''
    return html

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((IMG_SIZE, IMG_SIZE))
        img_array = np.array(img) / 255.0
        
        prediction = model.predict(np.array([img_array]), verbose=0)
        confidence = float(prediction[0][0]) * 100
        
        if confidence > 50:
            diagnosis = "PNEUMONIA DETECTED"
            icon = "⚠️"
            recommendation = "🚨 Urgent - Refer to Specialist Immediately"
            diagnosis_type = "pneumonia"
        else:
            confidence = (1 - float(prediction[0][0])) * 100
            diagnosis = "NORMAL"
            icon = "✅"
            recommendation = "✓ Routine Follow-up Recommended"
            diagnosis_type = "normal"
        
        return jsonify({
            'diagnosis': diagnosis,
            'confidence': round(confidence, 2),
            'icon': icon,
            'recommendation': recommendation,
            'type': diagnosis_type
        }), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("✓ Model loaded successfully!")
    print("🫁 Starting Pneumonia Detection Server...")
    print("📍 Server running at http://localhost:5000")
    print("🌐 Open http://localhost:5000 in your browser")
    app.run(debug=True, port=5000)
