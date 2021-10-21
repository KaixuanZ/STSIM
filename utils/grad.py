import matplotlib.pyplot as plt
import torch
import os

def viz_grad(grads, output_dir='tmp_viz'):
    '''
    Args:
        grads: []
    Returns
    '''
    grads = torch.stack(grads)  # list of grads ==> [iter, dim]
    for i in range(grads.shape[1]):
        y = grads[:,i].detach().cpu().numpy()
        x = range(grads.shape[0])
        plt.plot(x,y)
        plt.savefig(os.path.join(output_dir,'dim' + str(i) + '.png'))
        plt.close()

    for j in range(4):
        for i in range(14):   # show grad by subbands
            y = grads[:,i*4+j].detach().cpu().numpy()
            x = range(grads.shape[0])
            plt.plot(x,y,label='subband'+str(i))
        plt.legend()
        plt.savefig(os.path.join(output_dir,'feature' + str(j) + '.png'))
        plt.close()