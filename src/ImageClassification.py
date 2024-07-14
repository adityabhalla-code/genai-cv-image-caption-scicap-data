from tensorflow import keras
from tensorflow.keras import layers
import os
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
import numpy as np
from tensorflow.keras.utils import image_dataset_from_directory


def def_callbacks(filepath, mod_chk_mon = "val_loss", earlystop = 0 ):
    callback_list = []

    # Defualt callback
    callback_list.append(keras.callbacks.ModelCheckpoint(filepath,
                                         save_best_only = True,
                                         monitor=mod_chk_mon))

    if earlystop>0:
       callback_list.append(keras.callbacks.EarlyStopping(patience=earlystop))

    return callback_list

data_dir = "D:\SciCap\SciCap"

train_path = data_dir+'\Train'
validation_path = data_dir + '\Val'
test_path = data_dir + '\Test'
"""dim1 = []
dim2 = []
for image_file in os.listdir( train_path + '\WithoutSubFig\\' ):
    img = imread(train_path +'\WithoutSubFig\\'+image_file)
    d1,d2,colour_channels = img.shape
    dim1.append(d1)
    dim2.append(d2)

print("Mean across height of all dog images in train set is:",np.mean(dim1))
print("Mean across width of all dog images in train set is:",np.mean(dim2))"""


train_dataset = image_dataset_from_directory(
               train_path,
                image_size=(250, 250), # Resize the images to (180,180)
                batch_size=32,
                labels = 'inferred')

Val_dataset = image_dataset_from_directory(
               validation_path,
                image_size=(250,250), # Resize the images to (180,180)
                batch_size=32,
                labels = 'inferred')

Test_dataset = image_dataset_from_directory(
               test_path,
                image_size=(250,250), # Resize the images to (180,180)
                batch_size=32,
                labels = 'inferred')


inputs = keras.Input(shape=(250,250,3))
x = layers.Rescaling(1./255)(inputs)  
x = layers.Conv2D(filters=32, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=64, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=128, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.MaxPooling2D(pool_size=2)(x)
x = layers.Conv2D(filters=256, kernel_size=3, activation="relu")(x)
x = layers.Flatten()(x)
outputs = layers.Dense(1, activation="sigmoid")(x) 
model = keras.Model(inputs=inputs, outputs=outputs)

model.compile(loss="binary_crossentropy",   
                      optimizer="rmsprop",
                      metrics=["accuracy"])

epochs = 10

ModelFigSubFig    = model.fit(train_dataset,
                    epochs= epochs,
                    validation_data= Val_dataset,
                    callbacks= def_callbacks("D:\SciCap\convnet_FigSubFig.keras"))



