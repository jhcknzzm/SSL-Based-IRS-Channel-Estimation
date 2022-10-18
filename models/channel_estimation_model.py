import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

torch.manual_seed(1)
torch.cuda.manual_seed(1)
random.seed(0)
np.random.seed(0)


class SimSiam_Channel(nn.Module):
    def __init__(self, backbone=None):
        super().__init__()

        self.backbone = backbone
        print('____neural network structure')
        print(self.backbone)

        self.loss = torch.nn.MSELoss()

    def NMSE(self, output, gt):
        mse = F.mse_loss(output, gt, reduction='none')
        yhn = torch.sum(mse[:, 0, :, :], dim=[1, 2]) + torch.sum(mse[:, 1, :, :], dim=[1, 2])
        dfs = torch.sum(torch.pow(gt[:, 0, :, :], 2), dim=(1, 2)) + torch.sum(torch.pow(gt[:, 1, :, :], 2),
                                                                                dim=(1, 2))
        MSE = yhn / dfs
        return MSE

    def forward(self, method, x1, y, ls, std=0.55, DR=0):

        if method == 'supervise':

            f = self.backbone

            z1 = f(x1,ls)


            L = self.NMSE(z1, y).mean()

            return z1, z1,  {'loss': L}

        if 'sim' in method:

            f = self.backbone
            z1 = f(x1,ls)

            if 'x' in method:
                x1_n = x1 + torch.cuda.FloatTensor(x1.shape).normal_(mean=0, std=std)
                ls_dn = f(x1,x1_n)
                L = self.NMSE(ls_dn, x1).mean()


            return z1, z1,  {'loss': L}
