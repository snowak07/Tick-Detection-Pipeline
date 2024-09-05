#!/usr/bin/python3.10

import os
os.environ['MPLCONFIGDIR'] = '/var/tmp'
os.environ['YOLO_CONFIG_DIR'] = '/var/yolo_config'

import sys
sys.path.append('/var/pip')

import json
import math
from PIL import Image
import requests
from ultralytics import YOLO

CROP_SIZE = 224

##
# Saves image crop for each element in boxes param
##
def calculateCrop(img_path, boxes):
    image = Image.open(img_path)

    # Handle if the image is less than CROP_SIZE
    if image.width < CROP_SIZE or image.width < CROP_SIZE:
        return [{
            "start_x": 0,
            "start_y": 0,
            "end_x": image.width,
            "end_y": image.height
        }]

    crops = []

    # crop images based on boxes
    for box in boxes:
        x1, y1, x2, y2 = (box.xyxy).tolist()[0]

        center_x = ((x2 - x1) / 2) + x1
        center_y = ((y2 - y1) / 2) + y1

        start_x = calculateOffset(center_x, -(CROP_SIZE / 2))
        start_y = calculateOffset(center_y, -(CROP_SIZE / 2))

        if start_x == 0:
            end_x = calculateOffset(start_x, CROP_SIZE, image.width)
        else:
            end_x = calculateOffset(center_x, CROP_SIZE / 2, image.width)

            if end_x == image.width:
                start_x = calculateOffset(end_x, -CROP_SIZE)

        if start_y == 0:
            end_y = calculateOffset(start_y, CROP_SIZE, image.width)
        else:
            end_y = calculateOffset(center_y, CROP_SIZE / 2, image.width)

            if end_y == image.width:
                start_y = calculateOffset(end_y, -CROP_SIZE)

        crops.append({
            "start_x": start_x,
            "start_y": start_y,
            "end_x": end_x,
            "end_y": end_y
        })

    return crops

def calculateCenterCrop(img_path):
    image = Image.open(img_path)

    # Handle if the image is less than CROP_SIZE
    if image.width < CROP_SIZE or image.width < CROP_SIZE:
        return [{
            "start_x": 0,
            "start_y": 0,
            "end_x": image.width,
            "end_y": image.height
        }]

    center_x = image.width / 2
    center_y = image.height / 2

    start_x = calculateOffset(center_x, -(CROP_SIZE / 2))
    start_y = calculateOffset(center_y, -(CROP_SIZE / 2))
    end_x = calculateOffset(center_x, (CROP_SIZE / 2), image.width)
    end_y = calculateOffset(center_y, (CROP_SIZE / 2), image.height)

    return [{
        "start_x": start_x,
        "start_y": start_y,
        "end_x": end_x,
        "end_y": end_y
    }]

def calculateOffset(start_val, offset, limit = math.inf):
    result = start_val + offset

    if result < 0:
        result = 0

    if result > limit:
        result = limit

    return math.floor(result)

##
# main
##
def main():
    if len(sys.argv) < 3:
        return

    img_url = sys.argv[1]
    model_path = sys.argv[2]

    data = requests.get(img_url).content

    temp_file_name = "temp_img_file.jpg"
    file = open('/var/temp/' + temp_file_name, 'w+b')
    file.write(data)
    file.close()
    img_path = '/var/temp/' + temp_file_name

    if not os.path.isfile(img_path) or not os.path.isfile(model_path):
        print("Error: Invalid image_file or model: " + img_path + ", " + model_path)
        return

    # Load a YOLO model
    model = YOLO(model_path)

    # Perform object detection on an image using the model
    results = model.predict(img_path, verbose=False)

    # No objects found, default to center crop
    if results[0] == None:
        print(json.dumps(calculateCenterCrop(img_path)))
        return

    print(json.dumps(calculateCrop(img_path, results[0].boxes)))
    return

main()