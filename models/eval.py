import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils
import cv2
from PIL import Image
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy import ndimage
import math

model = load_model('models/mnistCNN.h5')

# fix random seed for reproducibility
seed = 7
np.random.seed(seed)

'''
-rwxrwxr-x 1 foglamp foglamp   16966 Apr  1 14:43 3digitdisplay-863.jpg
-rwxrwxr-x 1 foglamp foglamp   44844 Apr  1 14:43 digit-3.jpg
-rwxrwxr-x 1 foglamp foglamp   13367 Apr  1 14:43 digit-2.jpg
-rwxrwxr-x 1 foglamp foglamp    8287 Apr  1 14:43 digit-8.png
-rwxrwxr-x 1 foglamp foglamp   10544 Apr  1 14:43 digit-8-3.jpg
-rwxrwxr-x 1 foglamp foglamp   19650 Apr  1 14:43 digit-8-2.png
-rwxrwxr-x 1 foglamp foglamp    6157 Apr  1 14:43 digit-5.png
-rwxrwxr-x 1 foglamp foglamp   27882 Apr  1 14:43 digit-4.jpg
-rwxrwxr-x 1 foglamp foglamp    2388 Apr  1 14:43 digit-four.png
'''

IMG_SIZE = 28
#img_path = "./digit-four.png"
img_path = "./digit-2-ppd.jpg"
im = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
im = cv2.resize(255-im, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
(thresh, gray) = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
#im = im[:,:,1]
while np.sum(gray[0]) == 0:
    gray = gray[1:]

while np.sum(gray[:,0]) == 0:
    gray = np.delete(gray,0,1)

while np.sum(gray[-1]) == 0:
    gray = gray[:-1]

while np.sum(gray[:,-1]) == 0:
    gray = np.delete(gray,-1,1)

rows,cols = gray.shape

if rows > cols:
    factor = 20.0/rows
    rows = 20
    cols = int(round(cols*factor))
    gray = cv2.resize(gray, (cols,rows))
else:
    factor = 20.0/cols
    cols = 20
    rows = int(round(rows*factor))
    gray = cv2.resize(gray, (cols, rows))
    
colsPadding = (int(math.ceil((28-cols)/2.0)),int(math.floor((28-cols)/2.0)))
rowsPadding = (int(math.ceil((28-rows)/2.0)),int(math.floor((28-rows)/2.0)))
gray = np.lib.pad(gray,(rowsPadding,colsPadding),'constant')

def getBestShift(img):
    cy,cx = ndimage.measurements.center_of_mass(img)

    rows,cols = img.shape
    shiftx = np.round(cols/2.0-cx).astype(int)
    shifty = np.round(rows/2.0-cy).astype(int)

    return shiftx,shifty

def shift(img,sx,sy):
    rows,cols = img.shape
    M = np.float32([[1,0,sx],[0,1,sy]])
    shifted = cv2.warpAffine(img,M,(cols,rows))
    return shifted
shiftx,shifty = getBestShift(gray)
shifted = shift(gray,shiftx,shifty)
gray = shifted

im = gray/255.0
im2 = im
im = im / 255.0
#im = im.flatten()
im = im.reshape(1,784)
#print(img_path)

print(model.predict(im))
print(np.argmax(model.predict(im), axis=1))
# make prediction on test image using our trained model

prediction = model.predict(im, verbose=0)
#print(prediction)

pred = np.argmax(prediction, axis=1)

# display the prediction and image
print("[", img_path, "] I think the digit is", pred[0], " with", round(prediction[0][pred[0]] * 100, 2), "% probability")
#plt.imshow(im2, cmap=plt.get_cmap('gray'))
#plt.show()

