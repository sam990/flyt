import PIL 
# from keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
import time
import argparse
from torchvision.io import read_image
from torchvision.models import resnet50, ResNet50_Weights
import torch

parser = argparse.ArgumentParser(description ='Provide options to run total images and how many images to process at once on GPU.')
parser.add_argument('--batch-size', action="store", dest="batch_size", default=False, type=int, help='Provide the batch size, for e.g. --batch-size 50')
parser.add_argument('--total-images', action="store", dest="total_images", default=10000, type=int, help='Provide the total images to be recognised, for e.g. --total-images 100')
parser.add_argument('--image-name', action="store", dest="image_name", default="banana.jpg", type=str, help='Provide the image name to perform this experiment for e.g. --image-name banana.jpg')
arguments = parser.parse_args()

total_images = arguments.total_images
batch_size = arguments.batch_size
# image_name = arguments.image_name
image_name = "/home/ub-11/pramod/workload/banana.jpg"

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

# images = torch.from_numpy(images)
images = torch.cat([batch]*batch_size)
images = images.to(device)
# print(len(images))
#print(f'Length of array: {images}')
print("STARTING TIMER FOR PREDICTION")
# start = time.time()
total = 0
for i in range(0, total_images, batch_size):
    #print(f'running {i} times')
    predictions = model(images)
    total = len(predictions) + total
end = time.time()
#label = decode_predictions(predictions)
#print(f'predictions: {type()}')
class_id = predictions[0].argmax().item()
print(f'class id: {class_id}')
print(f'lenght of predictions[0] {len(predictions[0])}')
score = predictions[0][class_id].item()
category_name = weights.meta["categories"][class_id]
print(f"{category_name}: {100 * score}%")
print(f"total images predicted: {total}")
print(f'\n{end-start}')
