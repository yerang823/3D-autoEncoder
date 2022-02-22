import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import os

path='./test_mat/'
save_path='./test_mat_png/'
test_li=os.listdir(path)


for n in range(len(test_li)):
    vox=io.loadmat(path+test_li[n])
    vox2=np.array(vox['instance'])
    fig=plt.figure()
    ax=fig.gca(projection='3d')
    ax.voxels(vox2,edgecolor="red")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(save_path+test_li[n].split('.')[0]+'.png')
    #plt.show()
    print('Saved',save_path+test_li[n])
    
