# image recognition with YOLO
# Ruud Kapteijn 6-sep-21
# download pretrained model from https://pjreddie.com/darknet/yolo/

import cv2 as cv
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt

yolo = cv.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("coco.names", 'r') as f:
    classes = f.read().splitlines()

img = cv.imread('kat-natalie.jpg')
print("shape of image kat-natalie.jpg: " + str(img.shape))
imgResize = cv.resize(img, (320, 320))
# cv.imshow('output', imgResize)
# cv.waitKey(0)

blob = cv.dnn.blobFromImage(img, 1/255, (320,320), (0,0,0), swapRB=True, crop=False)
yolo.setInput(blob)
output_layers_names = yolo.getUnconnectedOutLayersNames()
layeroutput = yolo.forward(output_layers_names)
boxes = []
confidences = []
class_ids = []
for output in layeroutput:
    for detection in output:
        score = detection[5:]
        class_id = np.argmax(score)
        confidence = score[class_id]
        if confidence > 0.6:
            center_x = int(detection[0]*320)
            center_y = int(detection[0]*320)
            w = int(detection[0]*320)
            h = int(detection[0]*320)
            x = int(center_x-w/2)
            y = int(center_y-h/2)
            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

print(f"boxes: {len(boxes)}")
print(class_ids)

indexes = cv.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
font = cv.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(len(boxes), 3))

for i in indexes.flatten():
    x,y,w,h = boxes[i]
    label = str(classes[class_ids[i]])
    confi = str(round(confidences[i], 2))
    color = colors[i]
    cv.rectangle(imgResize, (x,y),(x+w, y+h), color, 1)
    cv.putText(imgResize, label+" "+confi, (x, y+20), font, 2, (255, 255, 255), 2)

# plt.imshow(imgResize)
cv.imshow('output', imgResize)
cv.waitKey(0)