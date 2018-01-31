import torch
import torch.nn as nn
from torch.autograd import Variable

import logging
import logzero
from logzero import logger

# logger setting
LOG_FORMAT = '[%(asctime)s %(levelname)s] %(message)s'
logzero.loglevel(logging.INFO)
logzero.formatter(logging.Formatter(LOG_FORMAT))
logzero.logfile('../log/contrastive-center.log')


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        self.use_cuda = use_cuda

    # may not work due to flowing gradient. change center calculation to exp moving avg may work.
    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        distances_same = distance_centers.gather(1, y.unsqueeze(1))
        intra_distances = distances_same.sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
               (inter_distances + epsilon) / 0.1
        logger.info('{},{},{}'.format(
            intra_distances.data[0], inter_distances.data[0], loss.data[0]))
        return loss

    def cuda(self, device_id=None):
        """Moves all model parameters and buffers to the GPU.

        Arguments:
            device_id (int, optional): if specified, all parameters will be
                copied to that device
        """
        self.use_cuda = True
        return self._apply(lambda t: t.cuda(device_id))


def test():
    ct = ContrastiveCenterLoss(2, 3)
    y = Variable(torch.LongTensor([0, 0, 0, 1, 1, 2]))
    feat = Variable(torch.randn(6, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()


if __name__ == '__main__':
    test()
