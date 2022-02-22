# =========== not completed ============= #


import numpy as np

path='../result/npy/test_pred_201016.npy'


result=np.load(path)
print(result.shape) # 970,32,32,32


#read test set name
name_path=r'../data/test_name_201016.txt'
f=open(name_path,'r')
name_li=f.readlines()
f.close()


#npy to raw
save_path='../result/raw/10_32_201016'
for i in range(result.shape[0]):
    #print(result[i].min(),result[i].max())
    
    #de-normalize with minmax 
    result_denorm=result[i]*((2800)-(-1024)) + (-1024)
    
    # save to raw
    f=open(save_path+'/'+name_li[i][:-4]+'raw','wb')
    buffer_=result_denorm.reshape(-1).astype('int16').tobytes()
    f.write(buffer_)
    f.close()
    


#f=open(save_path+'/'+'test.raw','wb')
#buffer_=result[0].reshape(-1).astype('float32').tobytes()
#f.write(buffer_)
#f.close()

        
        