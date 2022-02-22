from keras.models import load_model
import h5py
import numpy as np
#import mrcfile as mrc
import os
from data_process_noise import stack_mat

from keras import backend as K
K.clear_session()

'''
with h5py.File('object.hdf5', 'r') as f:
    train_data = f['train_mat'][...]
    val_data = f['val_mat'][...]
    test_data = f['test_mat'][...]
''' 
    
train_x,_,test_x,_,_,_=stack_mat()

print(train_x.shape)
#print(val_data.shape)
print(test_x.shape)

#train_num = train_data.shape[0]
#val_num = val_data.shape[0]
test_num = test_x.shape[0]
box_size = test_x.shape[1]

#train_data = train_data.reshape([-1, box_size, box_size, box_size, 1])
#val_data = val_data.reshape([-1, box_size, box_size, box_size, 1])
test_x = test_x.reshape([-1, box_size, box_size, box_size, 1])

#print(train_data.shape)
#print(val_data.shape)
print(test_x.shape)

autoencoder = load_model('../result/model/autoencoder_under16_201120.h5')
decoded_imgs = autoencoder.predict(test_x, batch_size=100)
decoded_imgs = decoded_imgs.reshape(test_num, box_size, box_size, box_size)
print("decoded imgs shape is:")
print(decoded_imgs.shape)

save_path='../result/npy/test_pred_under16_201120.npy'
np.save(save_path,decoded_imgs)
print('Saved at',save_path)



#=============================#
# npy to raw(result visualize)
#=============================#
result=decoded_imgs
print(result.shape) # 970,32,32,32


#read test set name
name_path=r'../data/test_name.txt'
f=open(name_path,'r')
name_li=f.readlines()
f.close()


# get npy(raw) and get min max (to de-normalize)
raw_path='../data/raw'


#npy to raw
save_path='../result/raw/3_norm_max1000'
for i in range(result.shape[0]):
    #print(result[i].min(),result[i].max())
    
    #de-normalize with minmax 
    result_denorm=result[i]*((1000)-(-1024)) + (-1024)
    
    # save to raw
    f=open(save_path+'/'+name_li[i][:-4]+'raw','wb')
    buffer_=result_denorm.reshape(-1).astype('int16').tobytes()
    f.write(buffer_)
    f.close()


'''
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# write back to hdf5 file
hdf5_file = h5py.File("reconstruction.hdf5", "w")
hdf5_file.create_dataset("recon_mat", decoded_imgs.shape, np.int8)
for i in range(len(decoded_imgs)):
    hdf5_file["recon_mat"][i] = decoded_imgs[i]

hdf5_file.close()
print('Reconstruction HDF5 file successfully created.')
'''