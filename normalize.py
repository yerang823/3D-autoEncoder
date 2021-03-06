import numpy as np
import os, cv2

path='../data/raw_32x32x32_NoduleMaskedRaw/'
savepath='../data/raw_32x32x32_normalize_re_NoduleMaskedRaw/'
li=os.listdir(path)



for i in range(len(li)):
  if li[i][-4:]=='.npy':
  	img=np.load(path+li[i])
  	#normimg=cv2.normalize(img,None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
  	normimg= (img-(-1024)) / (2800-(-1024))
  	normimg=normimg.astype('float32')
  	for k in range(32):
  		for j in range(32):
  			for ii in range(32):
  					if normimg[k,j,ii]<0:
  							normimg[k,j,ii]=0
  					if normimg[k,j,ii]>1:
  							normimg[k,j,ii]=1                        
    
   
  	print(i, "=================",normimg.min(), normimg.max())
  	
  	
  	np.save(savepath+li[i],normimg)
   
  	f=open( savepath+li[i][:-4]+'.raw','wb')
  	f.write(normimg.reshape(-1).tobytes())
  	f.close()
		
		     