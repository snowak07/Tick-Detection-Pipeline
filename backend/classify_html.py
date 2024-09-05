#!/usr/bin/python3.7

##
# Runner for classifying Ticks
#
# @copyright Center for Health Enhancement Systems Studies
##
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import sys
sys.path.append('/var/pip')

import argparse
from math import floor
import tensorflow as tf
import pandas as pd
from PIL import Image
import numpy as np
import cgi

import requests
import tempfile
import json

import logging
tf.get_logger().setLevel(logging.ERROR)

##
# Configuration Variables
##
version = "1.0.0"
img_shape = (224, 224, 3)
classes = ['Amblyomma americanum', 'Dermacentor variabilis', 'Ixodes scapularis']

##
# Main runner for program
#
# @return None
##
def main():
	global args
	args = readArguments()

	if (args.url.strip() != ""):
		image = downloadImage(args.url)

	elif (args.src_filepath.strip() != ""):
		image = loadImage(args.src_filepath)

	else:
		raise Exception("`url` or `src_filepath` must be supplied")

	model = loadModel(args.model_directory)
	classification_results = classify(model, image)

	print(json.dumps(classification_results))

##
# Classify image
#
# @param model			Model to use in classification
# @param image			Image to classify
# @param verbose		Indicates if we should be verbose when predicting
#
# @return any			Results
##
def classify(model, image, verbose = 0):
	global classes, args, version

	results = model.predict(image, batch_size=32, verbose=verbose)

	class_probability = np.amax(results, 1).tolist()
	rounded_class_probability = [round(100 * x, 2) for x in class_probability]
	class_ind = np.argmax(results, 1)
	predicted_class = [classes[i] for i in class_ind]
	class_probs = {}

	results = results.tolist()
	i = 0
	while i < len(results[0]):
		class_probs[classes[i]] = results[0][i]
		i += 1

	return {
		"version": version,
		"scripts_arguments": {
			"src_filepath": args.src_filepath,
			"model_directory": args.model_directory
		},
		"prediction": {
			"class": predicted_class[0],
			"probability": class_probability[0],
			"rounded_probability": rounded_class_probability[0]
		},
		"class_probabilities": class_probs
	}

##
# Return an image by url
#
# @param url			Url for image
#
# @return Image
##
def downloadImage(url):
	data = requests.get(url).content

	# Write the data to a temporary file
	tmp = tempfile.NamedTemporaryFile(mode='w+b', delete=False)
	tmp.write(data)

	# Open the image
	image = Image.open(tmp)
	img_resized = image.resize((img_shape[0], img_shape[1]))
	pixels = np.asarray(img_resized)  # convert image to array
	pixels = pixels.astype('float32')
	input = np.expand_dims(pixels, axis=0)  # adds batch dimension

	# Delete the temporary file
	tmp.close()

	return input

##
# Return an image by filepath
#
# @param path			Path to image
#
# @return Image
##
def loadImage(path):
	if (not os.path.exists(path)):
		raise Exception("Invalid input image file " + path)

	image = Image.open(path)
	img_resized = image.resize((img_shape[0], img_shape[1]))
	pixels = np.asarray(img_resized)  # convert image to array
	pixels = pixels.astype('float32')
	input = np.expand_dims(pixels, axis=0)  # adds batch dimension
	return input

##
# Load a model from its directory
#
# @param model_directory		Directory where model resides
#
# @return Model
##
def loadModel(model_directory):
	if (not os.path.isdir(model_directory)):
		raise Exception("Invalid model directory " + model_directory)

	return tf.keras.models.load_model(model_directory)

##
# Read arguments from command
#
# @return array
##
def readArguments():
	parser = argparse.ArgumentParser()

	parser.add_argument('-f',
						dest="src_filepath",
						type=str,
						default='',
						required=False,
						help="Path to image to classify. Images must be 224px by 224px.")

	parser.add_argument('-u',
						dest="url",
						type=str,
						default='',
						required=False,
						help="Url to image to classify. Images must be 224px by 224px.")

	parser.add_argument('-m',
						dest="model_directory",
						type=str,
						default="./model",
						required=False,
						help="Path to saved model directory")

	return parser.parse_args()

main()