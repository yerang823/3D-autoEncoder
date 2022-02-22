import os
from random import * 
import numpy as np

te_li=os.listdir('../data/test_y')
for num in range(len(te_li)):
  npy=np.zeros((32,32,32))
  for k in range(32):
    for j in range(32):
      for i in range(32):
          x=random() #0~1 float
          npy[i,j,k]=x
  np.save('../data/test_x/'+te_li[num],npy)
  print(num,"/",len(te_li), te_li[num])



tr_li=os.listdir('../data/train_y')
for num in range(len(tr_li)):
  npy=np.zeros((32,32,32))
  for k in range(32):
    for j in range(32):
      for i in range(32):
          x=random() #0~1 float
          npy[i,j,k]=x
  np.save('../data/train_x/'+tr_li[num],npy)
  print(num,"/",len(tr_li), tr_li[num])
  


val_li=os.listdir('../data/val_y')
for num in range(len(val_li)):
  npy=np.zeros((32,32,32))
  for k in range(32):
    for j in range(32):
      for i in range(32):
          x=random() #0~1 float
          npy[i,j,k]=x
  np.save('../data/val_x/'+val_li[num],npy)
  print(num,"/",len(val_li), val_li[num])