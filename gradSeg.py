import tensorflow as tf
#from tensorflow import keras
from PIL import Image
from keras_applications import imagenet_utils
from ICAU_5_5_edge import *
from ICAU import *
import PIL.Image
import numpy as np
import cv2
from keras_applications import imagenet_utils
import glob
import os

# Display
from IPython.display import Image, display
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def self_balanced_focal_loss(alpha=3, gamma=2.0):
    def loss(y_true, y_pred):
    
        y_pred = backend.softmax(y_pred, -1)
        cross_entropy = backend.categorical_crossentropy(y_true, y_pred)
        sample_weights = backend.max(backend.pow(1.0 - y_pred, gamma) * y_true, axis=-1)
        # class weights
        pixel_rate = backend.sum(y_true, axis=[1, 2], keepdims=True) / backend.sum(backend.ones_like(y_true),
                                                                                   axis=[1, 2], keepdims=True)
        class_weights = backend.max(backend.pow(backend.ones_like(y_true) * alpha, pixel_rate) * y_true, axis=-1)

        # final loss
        final_loss = class_weights * sample_weights * cross_entropy
        return backend.mean(backend.sum(final_loss, axis=[1, 2]))
    return loss

def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    #img = keras.preprocessing.image.load_img(img_path, target_size=size)
    img = tf.keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    #array = keras.preprocessing.image.img_to_array(img)
    array = tf.keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name,idx1,idx2):
    
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        # if pred_index is None:
        #     pred_index = tf.argmax(preds[0])
        # class_channel = preds[:, pred_index]

        class_channel = preds[0][idx1][idx2][1]

    
    grads = tape.gradient(class_channel, last_conv_layer_output)

    
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=1):
    # Load the original image
    img = tf.keras.utils.load_img(img_path)
    img = tf.keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)


def get_cord(img):
    points = []
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cont, hier = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in cont:
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])
    output = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    (numLabels, labels, stats, centroids) = output
    for i in range(1, numLabels):
        (cX, cY) = centroids[i]
        points.append((cX,cY))
        #p1_x = (cX+extLeft[0])/2.0
        #p1_y = (cY+extLeft[1])/2.0
        #p1_x = (cX+extRight[0])/2.0
        #p1_y = (cY+extRight[1])/2.0
        #p1_x = (cX+extTop[0])/2.0
        #p1_y = (cY+extTop[1])/2.0

        #p1_x = (cX+extBot[0])/2.0
        #p1_y = (cY+extBot[1])/2.0
        
        #points.append(extLeft)


model = keras.models.load_model('/ssd_scratch/cvit/varun/FCN-8s_based_on_VGG19_miou_0.803339_ep_313.h5',
                                 custom_objects={'loss':self_balanced_focal_loss(alpha=3, gamma=2.0),
                                 'InpaintContextAttentionUnit':InpaintContextAttentionUnit,
                                 'InpaintContextAttentionUnit5edge':InpaintContextAttentionUnit5edge})



def load_img(name):
  img = PIL.Image.open(name)
  return np.array(img)

def decode_one_hot(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)


#image directory
path = "/ssd_scratch/cvit/varun/selected/"
images = list(glob.glob(path+"*.png"))
img_size = (256, 448)
last_conv_layer_name = "block5_conv4"

for file in images:
  image = cv2.resize(load_img(file), dsize=(448,256))
  image = imagenet_utils.preprocess_input(image.astype(np.float32), data_format='channels_last', mode='torch')
  if np.ndim(image) == 3:
    image = np.expand_dims(image, axis=0)
  assert np.ndim(image) == 4
  prediction = model.predict(image)
  if np.ndim(prediction) == 4:
    pred_norm = np.squeeze(prediction, axis=0)
  pred_norm = decode_one_hot(pred_norm)
  rows, cols = np.where(pred_norm==1)
  if(len(rows)!=0): 
    p1 = (rows[0],cols[0])
    p2 = (rows[len(rows)-1],cols[len(cols)-1])
    p3 = (round((rows[0]+rows[len(rows)-1])/2),round((cols[0]+cols[len(cols)-1])/2))
    pairs = [p1,p2,p3]
    img_array = image
    model = model
    netmap = np.zeros((16,28))
    # Generate class activation heatmap
    for pair in pairs:
      heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name,pair[0],pair[1])
      netmap+=heatmap
    netmap/=len(pairs)
    save_and_display_gradcam(file,netmap, cam_path = "/home2/varungupta/Amazing-Semantic-Segmentation/diagram/"+os.path.basename(file))
  else:
    continue
