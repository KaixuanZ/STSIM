#import library numpy and give a short name
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd


# [dist11, dist12, dist13, dist21, dist22, dist23, dist31, dist32, dist33, original]
std = [0.41573971, 1.88561808, 1.33333333, 0.94280904, 1.6405359 , 1.66295884, 1.2862041 , 1.25707872, 0.0       , 0.41573971,
       1.96889391, 1.82574186, 1.93090524, 2.07869855, 1.89215404, 0.83147942, 1.49071198, 2.78886676, 1.68508343, 1.24721913,
       0.31426968, 0.68493489, 1.41421356, 1.34256066, 1.06574034, 1.05409255, 0.47140452, 1.49897084, 0.31426968, 0.0,
       1.03040206, 1.69967317, 1.47405546, 1.70692128, 1.39664501, 2.3570226 , 1.49897084, 0.66666667, 0.0       , 0.31426968,
       1.83249139, 2.3570226 , 2.00616334, 2.72618759, 2.10818511, 1.98761598, 2.53859104, 2.18298697, 0.31426968, 2.5819889,
       0.41573971, 0.81649658, 0.73702773, 0.7856742 , 0.68493489, 0.83147942, 0.56655772, 0.87488976, 0.49690399, 0.0,
       2.71256791, 1.77081972, 1.6405359 , 1.82574186, 2.39340658, 2.23330569, 1.5713484 , 1.74977953, 1.13311545, 1.2862041,
       0.66666667, 0.87488976, 0.87488976, 1.16534316, 0.7856742 , 0.49690399, 1.13311545, 1.36986978, 1.09994388, 0.68493489,
       0.91624569, 1.05409255, 0.68493489, 1.06574034, 0.94280904, 0.83147942, 0.83147942, 1.88561808, 0.31426968, 0.68493489,
       0.81649658, 0.47140452, 0.66666667, 0.83147942, 0.81649658, 0.87488976, 0.47140452, 0.81649658, 0.0       , 0.95581392,
       ]


# std = np.array([ 0.0275, 0.1246, 0.0881, 0.0623, 0.1084, 0.1098, 0.0850, 0.0830, 0.0,
#                 0.0807, 0.0749, 0.0792, 0.0853, 0.0776, 0.0341, 0.0611, 0.1144, 0.0691,
#                 0.0351, 0.0764, 0.1578, 0.1498, 0.1189, 0.1176, 0.0526, 0.1673, 0.0351,
#                 0.0946, 0.1560, 0.1353, 0.1567, 0.1282, 0.2163, 0.1376, 0.0612, 0.0,
#                 0.0843, 0.1085, 0.0923, 0.1255, 0.0970, 0.0915, 0.1168, 0.1005, 0.0145,
#                 0.0486, 0.0954, 0.0861, 0.0918, 0.0801, 0.0972, 0.0662, 0.1023, 0.0581,
#                 0.1395, 0.0910, 0.0843, 0.0939, 0.1230, 0.1148, 0.0808, 0.0900, 0.0583,
#                 0.0459, 0.0603, 0.0603, 0.0803, 0.0541, 0.0342, 0.0781, 0.0944, 0.0758,
#                 0.0618, 0.0711, 0.0462, 0.0719, 0.0636, 0.0561, 0.0561, 0.1272, 0.0212,
#                 0.0539, 0.0311, 0.0440, 0.0549, 0.0539, 0.0578, 0.0311, 0.0539, 0.0,
# ])

label = [[0, 0.038, 0.301, 0.369],[0, 0.234, 0.361, 0.335],[0, 0.181, 0.490, 0.595]]
pred = [[0, 0.050, 0.254, 0.327],[0, 0.317, 0.318, 0.335],[0, 0.198, 0.419, 0.611]]
outpath = "plot_dist_im01.eps"
#
def get_std(dir='/home/kaixuan/Downloads/scenes_data_tp'):
    mean = []
    std = []
    for i in range(10):
        path = os.path.join(dir, 'scene_'+str(i+1).zfill(2)+'n.xls')
        df = pd.read_excel(path, header=None)
        raw_data = df.iloc[:9,:10].to_numpy().astype(np.double)
        mean.append(raw_data.mean(0))
        std.append(raw_data.std(0))

    return np.stack(mean), np.stack(std)
mean, std = get_std()

import pdb;
pdb.set_trace()

def plot_fig6(label, pred, title, outpath):
    # dirs and pars
    fontsize = 20
    figsize_width = 4
    figsize_height = 3
    #data
    #numpy is an array of values
    x_array = np.linspace(0.5,3.5,4)
    #y1_array = [0.222, 0.778, 4.667, 5.667]
    #y2_array = [0.222, 3.667, 5.556, 5.111]
    #y3_array = [0.222, 2.889, 7.444, 9.000]
    # after inter-texture scaling

    plt.figure(figsize=(figsize_width,figsize_height),tight_layout=True)#,frameon=False)

    plt.xlabel('distortion level')
    plt.ylabel('distortion severity')
    plt.title(title)

    plt.plot(x_array,label[0], 'ro')
    plt.plot(x_array,label[1], 'go')
    plt.plot(x_array,label[2], 'bo')

    plt.plot(x_array,pred[0], 'r--')
    plt.plot(x_array,pred[1], 'g--')
    plt.plot(x_array,pred[2], 'b--')

    ax = plt.axes()

    ax.axis([-0.5, 5, -0.1, 1.1])

    plt.xticks([0.5,1.5,2.5,3.5], [0,1,2,3])

    plt.text(3.8, 0.2, 'rotations', color='red')
    plt.text(3.8, 0.3, 'shifts', color='green')
    plt.text(3.8, 0.4, 'warpings', color='blue')

    #This chunk of code puts the axes on the zero lines of x and y
    #The documentation: https://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    #Here I'm making the axes lines thicker

    width = 1.5

    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['top'].set_linewidth(width)

    #This part puts arrows on the ends of the axis
    #markers is the size of the arrows

    ax.plot((1), (0), ls="", marker=">", ms=5, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=5, color="k",transform=ax.get_xaxis_transform(), clip_on=False)
    plt.axis('off')
    plt.savefig(outpath,bbox_inches='tight')
    # plt.show()
    plt.close()

def plot_fig6_new(label, pred, std, title, outpath):
    # dirs and pars
    fontsize = 20
    figsize_width = 4
    figsize_height = 3
    #data
    #numpy is an array of values
    x_array = np.linspace(0.5,3.5,4)
    #y1_array = [0.222, 0.778, 4.667, 5.667]
    #y2_array = [0.222, 3.667, 5.556, 5.111]
    #y3_array = [0.222, 2.889, 7.444, 9.000]
    # after inter-texture scaling

    plt.figure(figsize=(figsize_width,figsize_height),tight_layout=True)#,frameon=False)

    plt.xlabel('distortion level')
    plt.ylabel('distortion severity')
    plt.title(title)

    # plt.plot(x_array,label[0], 'ro')
    # plt.plot(x_array,label[1], 'go')
    # plt.plot(x_array,label[2], 'bo')
    plt.errorbar(x_array, label[0], yerr=std[0], color='red', marker='o')
    plt.errorbar(x_array, label[1], yerr=std[1], color='green', marker='o')
    plt.errorbar(x_array, label[2], yerr=std[2], color='blue', marker='o')

    plt.plot(x_array,pred[0], 'r--')
    plt.plot(x_array,pred[1], 'g--')
    plt.plot(x_array,pred[2], 'b--')

    ax = plt.axes()

    ax.axis([-0.5, 5, -0.1, 1.1])

    plt.xticks([0.5,1.5,2.5,3.5], [0,1,2,3])

    plt.text(3.8, 0.2, 'rotations', color='red')
    plt.text(3.8, 0.3, 'shifts', color='green')
    plt.text(3.8, 0.4, 'warpings', color='blue')

    #This chunk of code puts the axes on the zero lines of x and y
    #The documentation: https://matplotlib.org/examples/pylab_examples/spine_placement_demo.html

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    #Here I'm making the axes lines thicker

    width = 1.5

    ax.spines['left'].set_linewidth(width)
    ax.spines['right'].set_linewidth(width)
    ax.spines['bottom'].set_linewidth(width)
    ax.spines['top'].set_linewidth(width)

    #This part puts arrows on the ends of the axis
    #markers is the size of the arrows

    ax.plot((1), (0), ls="", marker=">", ms=5, color="k",transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot((0), (1), ls="", marker="^", ms=5, color="k",transform=ax.get_xaxis_transform(), clip_on=False)
    plt.axis('off')
    plt.savefig(outpath,bbox_inches='tight')
    # plt.show()
    plt.close()


pred = np.array([0.0497, 0.2538, 0.3269, 0.3165, 0.3179, 0.3345, 0.1975, 0.4190, 0.6109,
        0.0386, 0.1571, 0.1287, 0.0812, 0.1605, 0.3893, 0.0618, 0.1742, 0.3215,
        0.1658, 0.3687, 0.5567, 0.4497, 0.6829, 0.8040, 0.2546, 0.6959, 1.0846,
        0.1367, 0.4320, 0.5814, 0.3409, 0.3858, 0.3688, 0.3263, 0.6124, 0.8710,
        0.0176, 0.0050, 0.0145, 0.0480, 0.0593, 0.1056, 0.1366, 0.2778, 0.5514,
        0.2470, 0.5970, 0.9689, 0.3920, 0.4735, 0.5245, 0.2811, 0.6919, 1.1017,
        0.0799, 0.1072, 0.2016, 0.1679, 0.2401, 0.3490, 0.0683, 0.1718, 0.4624,
        0.0519, 0.1696, 0.3264, 0.2277, 0.5084, 0.6242, 0.1372, 0.3784, 0.5250,
        0.0829, 0.2629, 0.4197, 0.2328, 0.3956, 0.4782, 0.1574, 0.3472, 0.4872,
        0.1021, 0.1225, 0.2647, 0.2839, 0.3584, 0.4544, 0.1495, 0.4817, 0.6776])

label = np.array([0.0380, 0.3014, 0.3692, 0.2336, 0.3615, 0.3347, 0.1807, 0.4899, 0.5945,
        0.0652, 0.1066, 0.1362, 0.1303, 0.2547, 0.3614, 0.0000, 0.1777, 0.2962,
        0.1260, 0.3839, 0.5576, 0.4464, 0.7312, 0.7808, 0.2232, 0.7312, 0.9921,
        0.1545, 0.4543, 0.6187, 0.3098, 0.3403, 0.3924, 0.3098, 0.6708, 0.8261,
        0.0000, 0.0000, 0.0164, 0.0573, 0.0000, 0.0327, 0.1227, 0.2782, 0.4091,
        0.1430, 0.7020, 0.9490, 0.3770, 0.5200, 0.5200, 0.2210, 0.8320, 1.0000,
        0.0896, 0.1539, 0.2113, 0.1792, 0.2819, 0.3458, 0.0896, 0.3073, 0.4227,
        0.0082, 0.1898, 0.3376, 0.2969, 0.4942, 0.5932, 0.1234, 0.3376, 0.5437,
        0.0552, 0.2525, 0.3550, 0.2207, 0.4420, 0.5049, 0.0792, 0.3706, 0.5997,
        0.0618, 0.1777, 0.2937, 0.2782, 0.4090, 0.4631, 0.0618, 0.4785, 0.5944])

std = np.array([ 0.0275, 0.1246, 0.0881, 0.0623, 0.1084, 0.1098, 0.0850, 0.0830, 0.0,
                0.0807, 0.0749, 0.0792, 0.0853, 0.0776, 0.0341, 0.0611, 0.1144, 0.0691,
                0.0351, 0.0764, 0.1578, 0.1498, 0.1189, 0.1176, 0.0526, 0.1673, 0.0351,
                0.0946, 0.1560, 0.1353, 0.1567, 0.1282, 0.2163, 0.1376, 0.0612, 0.0,
                0.0843, 0.1085, 0.0923, 0.1255, 0.0970, 0.0915, 0.1168, 0.1005, 0.0145,
                0.0486, 0.0954, 0.0861, 0.0918, 0.0801, 0.0972, 0.0662, 0.1023, 0.0581,
                0.1395, 0.0910, 0.0843, 0.0939, 0.1230, 0.1148, 0.0808, 0.0900, 0.0583,
                0.0459, 0.0603, 0.0603, 0.0803, 0.0541, 0.0342, 0.0781, 0.0944, 0.0758,
                0.0618, 0.0711, 0.0462, 0.0719, 0.0636, 0.0561, 0.0561, 0.1272, 0.0212,
                0.0539, 0.0311, 0.0440, 0.0549, 0.0539, 0.0578, 0.0311, 0.0539, 0.0,
])


outdir = 'tmp_figs'
if not os.path.isdir(outdir):
    os.mkdir(outdir)
for i in range(10):
    label_i = label[i*9:(i+1)*9]
    label_i = label_i.reshape(3,3)
    label_i = np.concatenate([np.zeros([3,1]), label_i],1)
    pred_i = pred[i*9:(i+1)*9]
    pred_i = pred_i.reshape(3,3)
    pred_i = np.concatenate([np.zeros([3,1]), pred_i],1)
    # std_i = std[i,:-1].reshape(3,3)
    std_i = std[i*9:(i+1)*9]
    std_i = std_i.reshape(3,3)
    # mean_i = mean[i]
    std_i = np.concatenate([np.zeros([3,1]), std_i],1)

    # import pdb;
    # pdb.set_trace()
    title = 'texture '+str(i+1)
    outpath = os.path.join(outdir, 'plot_dist_im' + str(i+1).zfill(2) + '.eps')
    # plot_fig6(label_i, pred_i, title, outpath)
    plot_fig6_new(label_i, pred_i, std_i, title, outpath)