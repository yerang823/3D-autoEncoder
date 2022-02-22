import matplotlib.pyplot as plt
from scipy import io
import numpy as np


path='result/test_pred.npy'
save_path='./test_pred_png/'

vox=np.load(path,allow_pickle=True)
print("vox.shape : ",vox.shape)
print('min, max, mean', vox[0].min(), vox[0].max(), vox[0].mean())


for n in range(10):#vox.shape[0]):
    '''
    vox2=np.array(vox[n])
    
    print(vox2.shape)
    print("min, max, avg =", vox2.min(), vox2.max(), vox2.mean())
    
    vox_cp=vox2.copy()
    
    for i in range(vox[n].shape[0]):
        for j in range(vox[n].shape[1]):
            for k in range(vox[n].shape[2]):
                if vox[n][i,j,k]!=0:
                  print(vox[n][i,j,k])

    '''
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    # ax.set_aspect('equal')
    ax.voxels(vox[n], edgecolor="red")
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.show()
    plt.savefig(save_path+str(n)+'.png')
    print(save_path+str(n)+'.png')
    
    