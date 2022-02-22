import os
from scipy import io
import numpy as np


def stack_mat():
  tr_path='./_train_mat/'
  te_path='./_test_mat/'
  val_path='./_val_mat/'
  
  tr=os.listdir(tr_path)
  te=os.listdir(te_path)
  val=os.listdir(val_path)
  

  train_vox=[]
  for i in range(len(tr)):
    img_tr=io.loadmat(tr_path+tr[i])
    
    img_tr=img_tr['instance']
    img_tr=np.pad(img_tr,1,'constant',constant_values=(0,0))
    img_tr=np.array(img_tr)
    train_vox.append(img_tr)
  train_vox=np.array(train_vox)
 
    
  test_vox=[]
  for i in range(len(te)):   
    img_te=io.loadmat(te_path+te[i])
    img_te=img_te['instance']
    img_te=np.pad(img_te,1,'constant',constant_values=(0,0))
    img_te=np.array(img_te)
    test_vox.append(img_te)
  test_vox=np.array(test_vox)

    
  val_vox=[]
  for i in range(len(val)):
    img_val=io.loadmat(val_path+val[i])
    
    img_val=img_val['instance']
    img_val=np.pad(img_val,1,'constant',constant_values=(0,0))
    img_val=np.array(img_val)
    val_vox.append(img_val)
  val_vox=np.array(val_vox) 
    
  return train_vox,test_vox,val_vox
