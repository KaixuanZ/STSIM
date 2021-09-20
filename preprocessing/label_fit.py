import sys
sys.path.append('..')
import os
import torch
import numpy as np
import pandas as pd
from utils.dataset import Dataset
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

image_dir = '/dataset/jana2012/'
label_file = 'label.xlsx'

df = pd.read_excel(os.path.join(image_dir,label_file), header=None)
label2 = df.iloc[12:22,:10].to_numpy().astype(np.double)

cross_file1 = 'scene_across1.xls'
cross_file2 = 'scene_across2.xls'
df = pd.read_excel(os.path.join(image_dir,cross_file1), header=None)
dis12 = df.iloc[:9,:10].to_numpy().astype(np.double)
dis12 = dis12.mean(0)
df = pd.read_excel(os.path.join(image_dir,cross_file2), header=None)
dis33 = df.iloc[:9,:10].to_numpy().astype(np.double)
dis33 = dis33.mean(0)

# scaling constant
c = []
for i in range(10):
    A = np.array([label2[1,i], label2[-2,i]])
    B = np.array([dis12[i], dis33[i]])
    c.append(np.dot(A,B)/np.dot(A,A))

c = np.array(c)
label_scaled = label2*c
label_scaled = label_scaled/label_scaled.max()
label_scaled[label_scaled<0] = 0
pd.DataFrame(label_scaled).to_csv('label_final.xls',index=False,header=False)
import pdb;pdb.set_trace()
tmp12 = np.multiply(label2[1],c)
plt.scatter(tmp12, dis12)
plt.xlabel('scaled value')
plt.ylabel('label')
plt.title('distortion12_PLCC_{:.3f}'.format(pearsonr(tmp12,dis12)[0]))
plt.savefig('tmp12.png')
plt.close()
tmp33 = np.multiply(label2[-2],c)

plt.scatter(tmp33, dis33)
plt.xlabel('scaled value')
plt.ylabel('label')
plt.title('distortion33_PLCC_{:.3f}'.format(pearsonr(tmp33,dis33)[0]))
plt.savefig('tmp33.png')
plt.close()