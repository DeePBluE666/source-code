import os
import tensorflow as tf
from keras.utils import plot_model
from matplotlib import pyplot as plt
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import gc
import keras
import random
import pandas as pd
import numpy as np
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Lambda, Embedding,constraints,\
Dropout, Activation,GRU,Bidirectional,Subtract, Permute, TimeDistributed, Reshape
from keras.layers import Conv1D,Conv2D,MaxPooling2D,GlobalAveragePooling1D,GlobalMaxPooling1D, MaxPooling1D, Flatten
from keras.layers import CuDNNGRU, CuDNNLSTM, SpatialDropout1D,Layer, initializers, regularizers
from keras.layers.merge import concatenate, Concatenate, Average, Dot, Maximum, Multiply, Subtract
from keras.models import Model
from keras.optimizers import RMSprop,Adam
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD
from keras import backend as K
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from keras.activations import softmax
from keras.utils import plot_model
from keras.layers import *
from keras.applications.imagenet_utils import _obtain_input_shape
from sklearn.metrics import log_loss
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from scipy.special import erfinv
from sklearn.metrics import accuracy_score
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.densenet import DenseNet121

def initialize_model(pretrain_model, num_unfreezed_layers, show = True):
    
    if pretrain_model == 'inceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'vgg16':
        base_model = VGG16(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'resnet':
        base_model = ResNet50(weights='imagenet', include_top=False)
        
    elif pretrain_model == 'densenet':
        base_model = DenseNet121(weights='imagenet', include_top=False)
        
    else:
        raise ValueError ('unexpected pretrain model %s'%pretrain_model)
        
    #add GlobelAvg2D layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    #add 2 FC layers 
    x = Dense(512, activation='relu')(x)

    predictions = Dense(6, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    
    #unfreeze layers
    if num_unfreezed_layers != 'all':
        for layer in base_model.layers[:-num_unfreezed_layers]:
            layer.trainable = False
    if show:
        print('*'*50)
        print('%s pretrain model is employed' %pretrain_model)
        print('unfreezed layer is %s'%num_unfreezed_layers)
        print('*'*50)
    #compile model
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model
	
def main_function(pretrain_model):
    
    if pretrain_model == 'inceptionV3':
        layer_list = [0, 60, 120, 180, 240, 'all']
        
    elif pertrain_model == 'vgg16':
        layer_list = [0, 3, 6, 9, 12, 'all']
        
    elif pretrain_model == 'resnet':
        layer_list = [0, 35, 70, 105, 140, 'all']
        
    elif pretrain_model == 'densenet':
        layer_list = [0, 85, 170, 255, 340, 'all']
        
    else:
        raise ValueError ('unexpected pretrain model %s'%pretrain_model)
        
#freeze ration = 0

base_model = DenseNet121(weights='imagenet', include_top=False)
x = base_model.output
x = BatchNormalization()(x)
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(6, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])                                      

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
        
train_generator = train_datagen.flow_from_directory(                 
        '/home/libo/dataset/ultraspnic/ult_data_train/',  
        target_size=(128, 128),                                       
        batch_size=32,
        class_mode='categorical')                                         


test_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = test_datagen.flow_from_directory(
        '/home/libo/dataset/ultraspnic/ult_data_test/',
        target_size=(128, 128),
        batch_size=32,
        class_mode='categorical')

early_stopping =EarlyStopping(monitor='val_acc', patience=5)
callbacks = [early_stopping,]
model.fit_generator(
        train_generator,
        steps_per_epoch=1024,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=256, callbacks=callbacks)
	
