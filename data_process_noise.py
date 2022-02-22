import os
from scipy import io
import numpy as np


def stack_mat():
  tr_x_path='../data/train_x/'
  te_x_path='../data/test_x/'
  val_x_path='../data/val_x/'

  tr_y_path='../data/train_y/'
  te_y_path='../data/test_y/'
  val_y_path='../data/val_y/'
    
  tr_x=os.listdir(tr_x_path)
  te_x=os.listdir(te_x_path)
  val_x=os.listdir(val_x_path)
  
  tr_y=os.listdir(tr_y_path)
  te_y=os.listdir(te_y_path)
  val_y=os.listdir(val_y_path)
  
  #============== x ================#
  
  train_x_vox=[]
  for i in range(len(tr_x)):
    #img_tr=io.loadmat(tr_path+tr[i])
    img_tr=np.load(tr_x_path+tr_x[i])
    #img_tr=img_tr['instance']
    #img_tr=np.pad(img_tr,1,'constant',constant_values=(0,0))
    #img_tr=np.array(img_tr)
    train_x_vox.append(img_tr)
  train_x_vox=np.array(train_x_vox)
    
    
  test_x_vox=[]
  f=open('../data/test_name.txt','w')
  for i in range(len(te_x)):
    #img_te=io.loadmat(te_path+te[i])
    img_te=np.load(te_x_path+te_x[i])
    #img_te=img_te['instance']
    #img_te=np.pad(img_te,1,'constant',constant_values=(0,0))
    #img_te=np.array(img_te)
    test_x_vox.append(img_te)
    f.write(te_x[i]+'\n')
  test_x_vox=np.array(test_x_vox)
  f.close()

    
 
  val_x_vox=[]
  for i in range(len(val_x)):
    #img_val=io.loadmat(val_path+val[i])
    img_val=np.load(val_x_path+val_x[i])
    #img_val=img_val['instance']
    #img_val=np.pad(img_val,1,'constant',constant_values=(0,0))
    #img_val=np.array(img_val)
    val_x_vox.append(img_val)
  val_x_vox=np.array(val_x_vox) 
  
  #============== y ================#
  
  train_y_vox=[]
  for i in range(len(tr_y)):
    #img_tr=io.loadmat(tr_path+tr[i])
    img_tr=np.load(tr_y_path+tr_y[i])
    #img_tr=img_tr['instance']
    #img_tr=np.pad(img_tr,1,'constant',constant_values=(0,0))
    #img_tr=np.array(img_tr)
    train_y_vox.append(img_tr)
  train_y_vox=np.array(train_y_vox)
    
    
  test_y_vox=[]
  #f=open('../data/test_name.txt','w')
  for i in range(len(te_y)):
    #img_te=io.loadmat(te_path+te[i])
    img_te=np.load(te_y_path+te_y[i])
    #img_te=img_te['instance']
    #img_te=np.pad(img_te,1,'constant',constant_values=(0,0))
    #img_te=np.array(img_te)
    test_y_vox.append(img_te)
    #f.write(te[i]+'\n')
  test_y_vox=np.array(test_y_vox)
  #f.close()

    
 
  val_y_vox=[]
  for i in range(len(val_y)):
    #img_val=io.loadmat(val_path+val[i])
    img_val=np.load(val_y_path+val_y[i])
    #img_val=img_val['instance']
    #img_val=np.pad(img_val,1,'constant',constant_values=(0,0))
    #img_val=np.array(img_val)
    val_y_vox.append(img_val)
  val_y_vox=np.array(val_y_vox) 
    
  return train_x_vox,train_y_vox,  test_x_vox,test_y_vox,  val_x_vox,val_y_vox
  
  
