from flask import Flask, request, jsonify
import PIL
import argparse
import time
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    batch_size = data['batch_size']

    total_images = 10000  # Default value, you can modify this if needed
    image_name = "/home/ub-11/pramod/workload/banana.jpg"  # Default image path, you can modify this if needed

    img = read_image(image_name)
    images = []

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    start = time.time()

    # Initialize model with the best available weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval().to(device)

    # Initialize the inference transforms
    preprocess = weights.transforms()
    batch = preprocess(img).unsqueeze(0)

    images = torch.cat([batch] * batch_size)
    images = images.to(device)

    total = 0
    for i in range(0, total_images, batch_size):
        predictions = model(images)
        total = len(predictions) + total
    
    end = time.time()

    # Calculate the time taken for execution
    elapsed_time = end - start

    return jsonify({"message": f"{elapsed_time} seconds"})

if __name__ == '__main__':
    app.run()

