import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd

def plot_hist(label, mask, output):
    bins = np.linspace(0, label.max(), 25)

    plt.hist(label[mask==1], bins, alpha=0.5, label='below threshold')
    plt.hist(label[mask==0], bins, alpha=0.5, label='above threshold')
    plt.legend(loc='upper right')
    if output[-1] == 'h':
        plt.title('Huib threshold')
    else:
        plt.title('Kaixuan threshold')
    plt.savefig(output)
    plt.close()

file1 = '/dataset/jana2012/label.xlsx'
df = pd.read_excel(file1, header=None)
label1 = df.iloc[:9, :10].to_numpy().astype(np.double)
label2 = df.iloc[12:21, :10].to_numpy().astype(np.double)
label3 = df.iloc[31:40, :10].to_numpy().astype(np.double)

file2 = '/dataset/jana2012/label_final.xlsx'
df = pd.read_excel(file2, header=None)
label4 = df.iloc[:9, :10].to_numpy().astype(np.double)

mask_pt_h = df.iloc[11:20, :10].to_numpy().astype(np.double)
mask_pt_kz = df.iloc[22:31, :10].to_numpy().astype(np.double)

plot_hist(label1, mask_pt_h, 'label1_h')
plot_hist(label1, mask_pt_kz, 'label1_kz')
plot_hist(label2, mask_pt_h, 'label2_h')
plot_hist(label2, mask_pt_kz, 'label2_kz')
plot_hist(label3, mask_pt_h, 'label3_h')
plot_hist(label3, mask_pt_kz, 'label3_kz')
plot_hist(label4, mask_pt_h, 'label4_h')
plot_hist(label4, mask_pt_kz, 'label4_kz')
import pdb;pdb.set_trace()