from imutils.video import FileVideoStream
import cv2
import time 
from imutils.video import VideoStream
import imutils
import os
import numpy as np
from random import shuffle
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import statistics 
from mtcnn.mtcnn import MTCNN
from keras.models import load_model
from mtcnn.mtcnn import MTCNN
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os


print("Loading model...")
facenet_model = load_model('./model/facenet_keras.h5')


def getLabel(face_image):
    face_array = np.asarray(face_image, dtype='float32')
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std 
    yhat = facenet_model.predict(np.expand_dims(face_array, axis=0))

    distances = []
    for train_vec, train_truth_label, _ in all_data:
        dist = np.linalg.norm(yhat - train_vec)
        distances.append((dist, train_truth_label))

    distances.sort()
    try:
        test_predicted_label = statistics.mode([l for _,l in distances[:5]]) 
    except statistics.StatisticsError as e:
        test_predicted_label = 'Confused'
    
    return test_predicted_label


def process(frame, required_size=(160, 160)):
    detector = MTCNN()
    width, height = frame.size
    pixels = np.asarray(frame)
    drawing = ImageDraw.Draw(frame)
    results = detector.detect_faces(pixels)
    #print(results)

    for face in results:
        #if face['confidence'] < 90:
        #    continue
        x1, y1, l, w = face['box']
        x2, y2 = x1 + l, y1 + w
        x1 = max(min(x1, width), 0)
        x2 = max(min(x2, width), 0)
        y1 = max(min(y1, height), 0)
        y2 = max(min(y2, height), 0)

        drawing.rectangle(((x1,y1),(x2,y2)), fill=None, width=2, outline='yellow')
        
        image = Image.fromarray(pixels[y1:y2, x1:x2]).resize(required_size)
        label = getLabel(image)
        drawing.text( (x1,y1-15), label)
    
    return frame


print("Loading training data...")
all_data = []

for personName in os.listdir('vectors'):
    for filename in os.listdir('vectors/' + personName):
        vec = np.load('vectors/' + personName + '/' + filename)
        all_data.append((vec, personName, 'dataset/' + personName + '/' + filename[:-4]))

print("Starting video stream...")
vs = FileVideoStream('output.avi').start()
fileStream = True


frameIndex = 0

while True:
    if fileStream and not vs.more():
        break
    pixels = vs.read()
    frameIndex += 1    
    print(frameIndex)
    
    if frameIndex % 30 == 0:
        pixels = imutils.resize(pixels, width=900)
        frame = Image.fromarray(pixels)

        #print(type(frame))
        frame = process(frame)
        pixels = np.asarray(frame)
        cv2.imshow("Frame", pixels)
        cv2.waitKey(1)
    

    