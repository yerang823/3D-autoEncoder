import h5py
from time import time
from keras.layers import Input
from keras.models import Model
from keras.layers.convolutional import Convolution3D, MaxPooling3D, UpSampling3D
from keras.layers import BatchNormalization
from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt
from data_process_noise import stack_mat
from keras import optimizers

import numpy as np
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

from keras import backend as K
K.clear_session()

'''
#with h5py.File('object.hdf5', 'r') as f:
with h5py.File('autoencoder_cp.h5', 'r') as f:
    train_data = f['train_mat'][...]
    val_data = f['val_mat'][...]
    test_data = f['test_mat'][...]
'''
train_x,train_y,test_x,test_y,val_x,val_y=stack_mat()

print(train_x.shape)
print(train_y.shape)
print(val_x.shape)
print(val_y.shape)
print(test_x.shape)
print(test_y.shape)


train_num = train_x.shape[0]
val_num = val_x.shape[0]
test_num = test_x.shape[0]
box_size = train_x.shape[1]

train_x = train_x.reshape([-1, box_size, box_size, box_size, 1])
val_x = val_x.reshape([-1, box_size, box_size, box_size, 1])
test_x = test_x.reshape([-1, box_size, box_size, box_size, 1])

train_y = train_y.reshape([-1, box_size, box_size, box_size, 1])
val_y = val_y.reshape([-1, box_size, box_size, box_size, 1])
test_y = test_y.reshape([-1, box_size, box_size, box_size, 1])





input_img = Input(shape=(32, 32, 32, 1))

x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(input_img)
x = MaxPooling3D((2, 2, 2), padding='same')(x)
x = BatchNormalization()(x)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(x)
encoded = MaxPooling3D((2, 2, 2), padding='same')(x)
# x = Convolution3D(120, (5, 5, 5), activation='relu', padding='same')(x)
# encoded = MaxPooling3D((2, 2, 2), padding='same', name='encoder')(x)

print("shape of encoded: ")
print(K.int_shape(encoded))

# x = Convolution3D(16, (5, 5, 5), activation='relu', padding='same')(encoded)
# x = UpSampling3D((2, 2, 2))(x)
x = Convolution3D(60, (5, 5, 5), activation='relu', padding='same')(encoded)
x = UpSampling3D((2, 2, 2))(x)
x = BatchNormalization()(x)
x = Convolution3D(30, (5, 5, 5), activation='relu', padding='same')(x)
x = UpSampling3D((2, 2, 2))(x)
decoded = Convolution3D(1, (5, 5, 5), activation='relu', padding='same')(x)
print("shape of decoded: ")
print(K.int_shape(decoded))

autoencoder = Model(input_img, decoded)
ada=optimizers.Adadelta(lr=0.01)
autoencoder.compile(optimizer=ada, loss='binary_crossentropy')
tensorboard = TensorBoard(log_dir="../logs/{}".format(time()))
autoencoder.fit(train_x, train_y,
              epochs=200,
              batch_size=100,
              validation_data=(val_x, val_y),
              callbacks=[tensorboard],
              use_multiprocessing=True)


autoencoder.save('../result/model/autoencoder_under16_201120.h5')
print("Training finished...")