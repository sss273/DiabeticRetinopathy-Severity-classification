import numpy as np
import pandas as pd

import os, cv2, base64, io, json
from PIL import Image
from flask import Flask, request

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import (Activation, Dropout, Flatten, Dense, GlobalMaxPooling2D,
						  BatchNormalization, Input, Conv2D, GlobalAveragePooling2D,concatenate,Concatenate)
from keras.callbacks import ModelCheckpoint
from keras import metrics
from keras.optimizers import Adam 
from keras import backend as K
import keras
from keras.models import Model

import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageOps
import cv2
from sklearn.utils import class_weight, shuffle
from keras.losses import binary_crossentropy, categorical_crossentropy
#from keras.applications.resnet50 import preprocess_input
from keras.applications.densenet import DenseNet121,DenseNet169
# from keras.applications.vgg16 import VGG16
import keras.backend as K
import tensorflow as tf
from sklearn.metrics import f1_score, fbeta_score, cohen_kappa_score
from keras.utils import Sequence
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.backend import set_session

app = Flask(__name__)

SIZE = 300 
NUM_CLASSES = 5

sess = tf.Session()

response_list = ["No DR", "Stage 1", "Stage 2", "Stage 3", "DR"]

def create_model(input_shape, n_out):
	input_tensor = Input(shape=input_shape)
	base_model = DenseNet121(include_top=False,weights=None,input_tensor=input_tensor)
	
	x = GlobalAveragePooling2D()(base_model.output)
	x = Dropout(0.5)(x)
	x = Dense(1024, activation='relu')(x)
	x = Dropout(0.5)(x)
	final_output = Dense(n_out, activation='softmax', name='final_output')(x)
	model = Model(input_tensor, final_output) 
	return model

model = create_model(input_shape=(SIZE,SIZE,3), n_out=NUM_CLASSES)
graph = tf.get_default_graph()
set_session(sess)
model.load_weights('best_model.h5')

def string_to_image(b64_string):
	print("inside string_to_image")
	image = Image.open(io.BytesIO(base64.b64decode(str(b64_string))))
	return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)

def predict(image):
	global sess
	global graph
	label_predict = 0
	image = cv2.resize(image, (SIZE, SIZE))
	X = np.array((image[np.newaxis])/255)
	with graph.as_default():
		set_session(sess)
		score_predict=((model.predict(X).ravel()*model.predict(X[:, ::-1, :, :]).ravel()*model.predict(X[:, ::-1, ::-1, :]).ravel()*model.predict(X[:, :, ::-1, :]).ravel())**0.25).tolist()
	label_predict = np.argmax(score_predict)
	return label_predict
	

@app.route('/detect', methods=['POST'])
def process_request():
	"""
	- Recieve request and process it on the basis of features in the request
	"""
	response = {}
	try:	
		print("Request recieved")
		print(request)
		image = string_to_image(request.form["image"])
		response['response'] = str(response_list[int(predict(image))])
	except Exception as e:
		response['response'] = str(e)

	return json.dumps(response)

if __name__ == "__main__":
	app.run(debug=True, host='0.0.0.0')
