from PIL import Image
import numpy as np
from config import img_size

def preprocess_image(img_path):
  img = Image.open(img_path).resize(img_size)
  img_array = np.array(img) / 255.0
  img_array = np.expand_dims(img_array, axis=0)
  return img_array, img

