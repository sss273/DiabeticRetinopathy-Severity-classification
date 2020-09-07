import keras
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os
import pandas as pd

SIresults = pd.DataFrame()

def prepare_image(i, file):
    img_path = '/media/New Volume/newdataset/IMAGES messidor2/'
    img = image.load_img(img_path +str(i)+'/'+file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return keras.applications.mobilenet_v2.preprocess_input(img_array_expanded_dims)

model = load_model('/home/PycharmProjects/FireDetection/diabetic_retinopathy/MODEL/saved_models/FT_MN210_DR_drop_0.1_lr_0.00015_BS_32_FTL_75.h5')


filelist, maxvalue_list,  max_index_list, class_label = ([] for i in range(4))
for i in range(5):
    for file in os.listdir("/media/New Volume/newdataset/IMAGES messidor2/"+str(i)):
        prediction = model.predict(prepare_image(i, file))
        maxvalue = np.argmax(prediction)
        max_index = np.amax(prediction)

        filelist.append(file)
        class_label.append(i)
        maxvalue_list.append(maxvalue)
        max_index_list.append(max_index)


SIresults['Filename'] = filelist
SIresults['class ground truth'] = class_label
SIresults['predictions class max'] = maxvalue_list
SIresults['prediction percent'] = max_index_list
SIresults.to_csv('testmesidor_results.csv', index=False)
