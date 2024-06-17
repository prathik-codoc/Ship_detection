from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load pretrained model
model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Transformation for the input image
transform = transforms.Compose([
    transforms.ToTensor()
])

def detect_ships(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        predictions = model(image)
    return predictions

@app.route('/detect', methods=['POST'])
def detect():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    image_file = request.files['image']
    try:
        image = Image.open(image_file).convert("RGB")
    except Exception as e:
        return jsonify({'error': f'Error opening image: {str(e)}'}), 400

    predictions = detect_ships(image)
    print(f"Predictions: {predictions}")
    result = [{'bbox': box.tolist(), 'score': score.item()} 
              for box, score in zip(predictions[0]['boxes'], predictions[0]['scores']) if score > 0.5]
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
