import PIL 
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from keras.applications.imagenet_utils import decode_predictions 
import matplotlib.pyplot as plt 
import numpy as np 
from tensorflow.keras.applications.resnet50 import ResNet50 
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications import vgg16
import time
import argparse
from torchvision.io import read_image
from torchvision.models import  ResNet50_Weights
import torch

start = time.time()
parser = argparse.ArgumentParser(description ='Provide options to run total images and how many images to process at once on GPU.')
parser.add_argument('--batch-size', action="store", dest="batch_size", default=False, type=int, help='Provide the batch size, for e.g. --batch-size 50')
parser.add_argument('--total-images', action="store", dest="total_images", default=10000, type=int, help='Provide the total images to be recognised, for e.g. --total-images 100')
parser.add_argument('--image-name', action="store", dest="image_name", default="/home/ub-11/pramod/image_recognition/banana.jpg", type=str, help='Provide the image name to perform this experiment for e.g. --image-name banana.jpg')
arguments = parser.parse_args()

total_images = arguments.total_images
batch_size = arguments.batch_size
#image_name = arguments.image_name
image_name = "/home/ub-11/pramod/workload/banana.jpg"
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
images = []
filename = image_name
original = load_img(filename, target_size = (224, 224))
'''
image = read_image(image_name)
weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()
batch = preprocess(image).unsqueeze(0)
images_pytorch = torch.cat([batch]*batch_size)
'''
#print('PIL image size',original.size)
#plt.imshow(original)
#plt.show()
numpy_image = img_to_array(original)
#plt.imshow(np.uint8(numpy_image))
#print('numpy array size',numpy_image.shape)
image_batch = np.expand_dims(numpy_image, axis = 0)
#print('image batch size', image_batch.shape)
#processed_image = resnet50.preprocess_input(image_batch.copy())
processed_image = vgg16.preprocess_input(image_batch.copy())
#resnet_model = resnet50.ResNet50(weights = 'imagenet')
vgg16_model = tf.keras.applications.VGG16(weights="imagenet")
#xception_model = tf.keras.applications.xception.Xception(weights='imagenet')


for i in range(0, batch_size):
    images.append(processed_image)
#print("Images: "+str(np.vstack(images)))
# print("STARTING TIMER FOR PREDICTION")
total = 0

for i in range(0, total_images, batch_size):
    #print(f'running {i} times')
    predictions = vgg16_model.predict(np.vstack(images), verbose = 0)
    #predictions = xception_model.predict(np.vstack(images), verbose = 0)
    #predictions = vgg16_model.predict(np.vstack(images), verbose = 0)
    total = len(predictions) + total
end = time.time()
label = decode_predictions(predictions)
# print(f"Total images predicted is: {total}, and prediction is: {label}")
print(f'\n{end-start}')
