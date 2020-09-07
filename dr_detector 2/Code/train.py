import os
from glob import glob
from skimage.io import imread

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

%matplotlib inline 
import preprocess

# Importing models
import resnet50
import vgg16
import inception_v3


# Tensorflow, Sklearn & Keras related imports
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from keras import backend as K
from keras.utils.np_utils import to_categorical
from keras.applications.inception_v3 import preprocess_input

from keras.applications.vgg16 import VGG16
# from keras.applications.inception_resnet_v2 import InceptionResNetV2 as PTModel
from keras.applications.inception_v3 import InceptionV3 as PTModel
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.layers import BatchNormalization
from keras.metrics import top_k_categorical_accuracy
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau

# Images 

root_directory = '../input/diabetic-retinopathy-detection'
images = glob('../input/diabetic-retinopathy-detection/*.jpeg')
labels = ('../input/diabetic-retinopathy-detection/trainLabels.csv')
weight_path="dr_weights.best.hdf5"

# Utility Functions

check_path = os.path.exists
split_name = lambda x: x.split('_')[0]
image_path = lambda x: os.path.join(root_directory,'{}.jpeg'.format(x))
get_level  = lambda x: to_categorical(x,1+dataset['level'].max())
check_left_or_right = lambda x: 1 if x.split('_')[-1]=='left' else 0
resample_data = lambda x: x.sample(100, replace = True)

labels_df = pd.read_csv(labels)
dataset = pd.DataFrame()

dataset['image'] = labels_df['image']
dataset['level'] = labels_df['level']

dataset['p_id'] = labels_df['image'].map(split_name)
dataset['path'] = labels_df['image'].map(image_path)

dataset['path_exists'] = dataset['path'].map(check_path)
dataset['l_r_eye'] = dataset['image'].map(check_left_or_right)
dataset['level_cat'] = dataset['level'].map(get_level)

# Removing nulls and rows where path does not exist (since we are working on a subset)
dataset.dropna(inplace = True)
dataset = dataset[dataset['path_exists']]

# Splitting dataset on the basis of IDs (train+validation)
train_ids, valid_ids = train_test_split(dataset['p_id'], test_size = 0.25, random_state = 2018, stratify = dataset['level'])

# Splitting dataframe on the basis of train & validation IDs
raw_train_df = dataset[dataset['p_id'].isin(train_ids)]
valid_df = dataset[dataset['p_id'].isin(valid_ids)]

print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])

train_df = raw_train_df.groupby(['level', 'l_r_eye']).apply(resample_data).reset_index(drop = True)
print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])

IMAGE_SIZE = (512, 512)
BATCH_SIZE = 64

core_idg = preprocess.augmentor(out_size = IMAGE_SIZE, color_mode = 'rgb', vertical_flip = True,crop_probability=0.0, batch_size = BATCH_SIZE) 
valid_idg = preprocess.augmentor(out_size = IMAGE_SIZE, color_mode = 'rgb', crop_probability=0.0, horizontal_flip = False, vertical_flip = False, random_brightness = False,
                         random_contrast = False,random_saturation = False,random_hue = False,rotation_range = 0,batch_size = BATCH_SIZE)

# Generating data flow to generate data
train_gen = preprocess.flow_df(core_idg, train_df, path_col = 'path', y_col = 'level_cat')
valid_gen = preprocess.flow_df(valid_idg, valid_df, path_col = 'path', y_col = 'level_cat')

t_x, t_y = next(train_gen)

def top_2_accuracy(in_gt, in_pred):
    return top_k_categorical_accuracy(in_gt, in_pred, k=2)

checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only = True)
reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=5, min_lr=0.0001)
early = EarlyStopping(monitor="val_loss", mode="min", patience=6) # probably needs to be more patient, but kaggle time is limited

callbacks_list = [checkpoint, early, reduceLROnPlat]

#classifier = create_attention_w_inceptionV3_model()
# classifier = create_attention_w_vgg16_model()
classifier = resnet50.create_attention_w_resnet50_model()

classifier.fit_generator(train_gen, steps_per_epoch = train_df.shape[0]//BATCH_SIZE,validation_data = valid_gen, validation_steps = valid_df.shape[0]//BATCH_SIZE,
                            epochs = 25,callbacks = callbacks_list,workers = 0,use_multiprocessing=True, max_queue_size = 0)

classifier.load_weights(weight_path)
classifier.save('resnet50_model.h5')

##### create one fixed dataset for evaluating
from tqdm import tqdm_notebook
# fresh valid gen
valid_gen = flow_df(valid_idg, valid_df, path_col = 'path',y_col = 'level_cat') 

vbatch_count = (valid_df.shape[0]//BATCH_SIZE-1)
out_size = vbatch_count*BATCH_SIZE
test_X = np.zeros((out_size,)+t_x.shape[1:], dtype = np.float32)
test_Y = np.zeros((out_size,)+t_y.shape[1:], dtype = np.float32)

for i, (c_x, c_y) in zip(tqdm_notebook(range(vbatch_count)), valid_gen):
    j = i*BATCH_SIZE
    test_X[j:(j+c_x.shape[0])] = c_x
    test_Y[j:(j+c_x.shape[0])] = c_y

pred_Y = classifier.predict(test_X, batch_size = 32, verbose = True)
pred_Y_cat = np.argmax(pred_Y, -1)
test_Y_cat = np.argmax(test_Y, -1)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(test_Y_cat, pred_Y_cat)))
print(classification_report(test_Y_cat, pred_Y_cat))