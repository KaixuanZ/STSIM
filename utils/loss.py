import torch

def PearsonCoeff(X, Y, mask):
    '''
    Args:
        X: [N, 1] neural prediction for one batch
        Y: [N] label
        mask: [N] indicator of correspondent class, e.g. [0,0,1,1] ,means first two samples are class 0, the rest two samples are class 1
    Returns: Borda's rule of pearson coeff between X&Y, the same as using numpy.corrcoef()
    '''
    coeff = 0
    N = set(mask.detach().cpu().numpy())
    for i in N:
        X1 = X[mask == i, 0].double()
        X1 = X1 - X1.mean()
        X2 = Y[mask == i].double()
        X2 = X2 - X2.mean()

        nom = torch.dot(X1, X2)
        denom = torch.sqrt(torch.sum(X1 ** 2) * torch.sum(X2 ** 2))

        coeff += torch.abs(nom / denom)
    return coeff / len(N)


if __name__ == '__main__':
    N = 100
    X = torch.rand(N,1)
    Y = torch.rand(N)
    mask = []
    M = 100
    for i in range(N//M):
        mask += [i**2]*M
    mask = torch.tensor(mask)
    print(PearsonCoeff(X,Y,mask))

    import numpy as np
    def Borda_rule(pred, label, mask):
        '''
        expectation of Pearson's corr over all textures
        :param pred: values predicted by the metric
        :param label: ground truth label
        :param N: number of distortions per texture
        :return: Pearson's corr with Borda's rule
        '''
        coeffs = 0
        pred = pred.cpu().numpy()
        label = label.cpu().numpy()
        N = set(mask.detach().cpu().numpy())
        for i in N:
            corr = np.corrcoef(pred[mask == i, 0], label[mask == i])[0, 1]
            coeffs += np.abs(corr)
        return coeffs / len(N)

    print(Borda_rule(X,Y,mask))
    import pdb;
    pdb.set_trace()