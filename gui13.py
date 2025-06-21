from flask import Flask, render_template_string, request
from PIL import Image
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
import easyocr
from transformers import pipeline
import os
import base64
from io import BytesIO

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load EasyOCR
reader = easyocr.Reader(['en'])

# Load text emotion classifier
text_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion")

# Define image emotion model
class EmotionNet(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionNet, self).__init__()
        self.model = resnet18(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

# Emotion labels from FER dataset
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
model = EmotionNet()
model.eval()  # Model is not trained, for demo purposes only

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

HTML = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Meme Analyzer</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(135deg, #f093fb, #f5576c);
            color: #fff;
            margin: 0;
            padding: 0;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .container {
            background: rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 40px;
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            text-align: center;
            max-width: 600px;
            width: 90%;
        }
        h1 {
            color: #fff;
            margin-bottom: 30px;
            font-weight: 600;
            letter-spacing: 1px;
        }
        input[type="file"] {
            padding: 12px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.1);
            color: #fff;
            width: calc(100% - 24px);
            box-sizing: border-box;
        }
        input[type="file"]::file-selector-button {
            background-color: #fff;
            color: #f5576c;
            border: none;
            padding: 10px 15px;
            border-radius: 6px;
            cursor: pointer;
            transition: background-color 0.3s ease;
        }
        input[type="file"]::file-selector-button:hover {
            background-color: #eee;
        }
        button[type="submit"] {
            background-color: #fff;
            color: #000; /* Changed to black */
            border: none;
            padding: 12px 25px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: background-color 0.3s ease, color 0.3s ease;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        button[type="submit"]:hover {
            background-color: #000; /* Changed to black */
            color: #fff;
        }
        .result-container {
            margin-top: 40px;
            background: rgba(255, 255, 255, 0.1);
            padding: 30px;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.1);
            text-align: left;
        }
        .result-container h2 {
            color: #fff;
            margin-top: 0;
            margin-bottom: 20px;
            font-weight: 500;
            border-bottom: 2px solid rgba(255, 255, 255, 0.2);
            padding-bottom: 10px;
        }
        .result-item {
            margin-bottom: 15px;
            font-size: 16px;
        }
        .result-item strong {
            font-weight: 600;
            color: #000; /* Changed to black */
        }
        .meme-image {
            margin-top: 20px;
            border-radius: 10px;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.2);
            max-width: 100%;
            height: auto;
        }
        .error {
            color: #ff4d4d;
            margin-top: 20px;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Meme Analyzer</h1>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="meme" accept="image/*" required><br>
            <button type="submit">Analyze Meme</button>
        </form>

        {% if image %}
        <img src="{{ image }}" alt="Uploaded Meme" class="meme-image"><br>
        <div class="result-container">
            <h2>Analysis Results</h2>
            <p class="result-item"><strong>📝 Extracted Text:</strong> {{ extracted_text }}</p>
            <p class="result-item"><strong>💬 Text Emotion:</strong> {{ text_emotion.label }} (Score: {{ '%.2f' % text_emotion.score }})</p>
            <p class="result-item"><strong>🖼 Image Emotion:</strong> {{ image_emotion.label }} (Score: {{ '%.2f' % image_emotion.score }})</p>
            <p class="result-item"><strong>🎭 Mixed Emotion:</strong> {{ mixed_emotion }}</p>
        </div>
        {% endif %}

        {% if error %}
        <p class="error">{{ error }}</p>
        {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)

@app.route('/', methods=['POST'])
def analyze_meme():
    if 'meme' not in request.files:
        return render_template_string(HTML, error='No file part')
    file = request.files['meme']
    if file.filename == '':
        return render_template_string(HTML, error='No selected file')
    if file:
        try:
            # Save the uploaded image temporarily
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            img = Image.open(filepath).convert("RGB")

            # --- TEXT EMOTION ---
            gray_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
            temp_path = os.path.join(app.config['UPLOAD_FOLDER'], "gray.png")
            cv2.imwrite(temp_path, gray_img)
            text_result = reader.readtext(temp_path, detail=0)
            extracted_text = ' '.join(text_result).strip()

            if extracted_text:
                text_emotion = text_classifier(extracted_text)[0]
            else:
                text_emotion = {'label': 'No Text', 'score': 0}

            # --- IMAGE EMOTION ---
            input_tensor = transform(img).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_idx = torch.argmax(probs, dim=1).item()
                image_emotion = {
                    "label": emotion_labels[top_idx],
                    "score": round(probs[0][top_idx].item(), 2)
                }

            mixed_emotion = f"{text_emotion['label']} + {image_emotion['label']}"

            # Encode the image to base64 for display in HTML
            img_buffer = BytesIO()
            img.save(img_buffer, format="PNG")
            img_str = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
            image_data_uri = f"data:image/png;base64,{img_str}"

            return render_template_string(HTML,
                                   image=image_data_uri,
                                   extracted_text=extracted_text if extracted_text else "No Text",
                                   text_emotion=text_emotion,
                                   image_emotion=image_emotion,
                                   mixed_emotion=mixed_emotion)
        except Exception as e:
            return render_template_string(HTML, error=f'Error processing image: {e}')
        finally:
            # Clean up temporary files
            if os.path.exists(filepath):
                os.remove(filepath)
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)