import os
import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.applications import imagenet_utils
from keras.applications import mobilenet_v2
from keras.applications.mobilenet_v2 import preprocess_input, MobileNetV2
from keras.optimizers import Adam
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, TensorBoard
from keras import callbacks
from keras.utils import multi_gpu_model
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# GPU start

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

# paths
train_path = '/media//New Volume//Final dataset/train'
valid_path = '/media//New Volume//Final dataset/val'
model_path = '/home//PycharmProjects/FireDetection/diabetic_retinopathy/MODEL/saved_models/'
callbacks_path = '/home//PycharmProjects/FireDetection/diabetic_retinopathy/MODEL/saved_models/logs/'

# Hyper Parameters

learning_rate = [0.00015] #[0.01, 0.001, 0.0001]
Batch_size = [32] #[64, 128, 256]
layers = range(75, 80, 5)

# learning_rate = [0.001, 0.0001, 0.002, 0.0002]
# Batch_size = [32, 64, 128, 256]
# # optimizers = [adam, adam_amsgrad] - ams grad converges faster and shows same behavior
# layers = range(75, 125, 5)
epochs = 20  # 20 or 25 will be sufficient initially 50 long time
dropouts = range(10, 15, 5)

# loading mobile net

mobilenet = MobileNetV2(input_shape=None, alpha=1.4, include_top=False, weights='imagenet', input_tensor=None,
                        pooling=None)


def Mobilenet(dropout=0.1):
    mobilenet.trainable = False
    x = mobilenet.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)  # model can learn more complex functions and classify for better results
    x = Dense(1024, activation='relu')(x)  # dense layer 2
    x = Dense(512, activation='relu')(x)  # dense layer 3
    x = Dropout(dropout)(x)
    preds = Dense(5, activation='softmax')(x)  # final layer with softmax activation
    mobilenetmodel = Model(inputs=mobilenet.input, outputs=preds)
    return mobilenetmodel


# Data Pre processing

def imagegenerator(path):
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    image_generator = datagen.flow_from_directory(path, target_size=(224, 224),
                                                  color_mode='rgb',
                                                  batch_size=32,
                                                  class_mode='categorical',
                                                  shuffle=True)

    return image_generator


train_data = imagegenerator(train_path)
val_data = imagegenerator(valid_path)
print(len(train_data), len(val_data))

for dropout in dropouts:
    for lr_rate in learning_rate:
        for bs in Batch_size:
            for lyr in layers:

                # cuda.select_device(0)
                # cuda.select_device(1)
                if lr_rate == 0.01:
                    epochs = 30
                else:
                    epochs = epochs

                model = Mobilenet(dropout / 100)

                adam_amsgrad = Adam(lr=lr_rate, amsgrad=True)
                model.compile(optimizer=adam_amsgrad, loss='categorical_crossentropy', metrics=['accuracy'])
                model.summary()

                # Model path
                savedname = "MN210_DR_drop_" + str(dropout / 100) + "_lr_" + str(lr_rate) + "_BS_" + str(
                    bs) + "_FTL_" + str(lyr)
                modelname = savedname + ".h5"
                modelpath = model_path + modelname
                print(savedname)

                # training

                # callback filenames
                modelckpt = callbacks_path + savedname + ".ckpt"
                csvpath = callbacks_path + savedname + ".csv"
                tbpath = callbacks_path + "./" + savedname

                # callbacks

                cp_callback = ModelCheckpoint(modelckpt, monitor='val_loss', verbose=0, save_best_only=False,
                                              save_weights_only=False, mode='auto', period=1)
                reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=0.00001)
                early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
                                               baseline=None, restore_best_weights=False)
                csv_logger = CSVLogger(csvpath, separator=',', append=True)
                tensorboard = TensorBoard(log_dir=tbpath, histogram_freq=0, write_graph=True,
                                          write_grads=False, write_images=False, batch_size=bs,
                                          embeddings_freq=0, embeddings_layer_names=None,
                                          embeddings_metadata=None, embeddings_data=None,
                                          update_freq='epoch')

                initial_epochs = epochs
                batch_size = bs
                steps_per_epoch = train_data.n // batch_size
                validation_steps = val_data.n // batch_size

                try:
                    history = model.fit_generator(train_data,
                                                  steps_per_epoch=steps_per_epoch,
                                                  epochs=initial_epochs,
                                                  workers=4,
                                                  validation_data=val_data,
                                                  validation_steps=validation_steps,
                                                  callbacks=[csv_logger, cp_callback, early_stopping, reduce_lr,
                                                             tensorboard])
                    model.save(modelpath)
                except KeyboardInterrupt:
                    model.save(modelpath)
                    break

                # fine - tuning layers

                mobilenet.trainable = True
                fine_tune_at = lyr
                for layer in mobilenet.layers[:fine_tune_at]:
                    layer.trainable = False

                adam_amsgrad = Adam(lr=lr_rate, amsgrad=True)
                model.compile(optimizer=adam_amsgrad, loss='categorical_crossentropy', metrics=['accuracy'])
                model.summary()

                begin_epochs = early_stopping.stopped_epoch

                if 0 < begin_epochs < initial_epochs:
                    begin_epochs = begin_epochs + 1  # begin_epochs
                else:
                    begin_epochs = initial_epochs

                final_epochs = begin_epochs + epochs
                tunedpath = model_path + "FT_" + savedname + ".h5"

                try:
                    history_tuned = model.fit_generator(train_data,
                                                        steps_per_epoch=steps_per_epoch,
                                                        epochs=final_epochs,
                                                        workers=4,
                                                        validation_data=val_data,
                                                        validation_steps=validation_steps,
                                                        initial_epoch=begin_epochs,
                                                        callbacks=[csv_logger, cp_callback, early_stopping, reduce_lr,
                                                                   tensorboard])
                    model.save(tunedpath)
                except KeyboardInterrupt:
                    model.save(tunedpath)
                    break
                del model
                # cuda.close()
                print('files saved')
