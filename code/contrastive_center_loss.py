import torch
import torch.nn as nn
from torch.autograd import Variable


class ContrastiveCenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(ContrastiveCenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        self.use_cuda = use_cuda

    def forward(self, y, hidden):
        batch_size = hidden.size()[0]
        expanded_centers = self.centers.expand(batch_size, -1, -1)
        expanded_hidden = hidden.expand(self.num_classes, -1, -1).transpose(1, 0)
        distance_centers = (expanded_hidden - expanded_centers).pow(2).sum(dim=-1)
        intra_distances = distance_centers.index_select(dim=0, index=y.long()).sum()
        inter_distances = distance_centers.sum().sub(intra_distances)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0 / batch_size) * intra_distances / \
               (inter_distances + epsilon)
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
    ct = ContrastiveCenterLoss(2, 2)
    y = Variable(torch.LongTensor([0, 0, 0, 1]))
    feat = Variable(torch.zeros(4, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()


if __name__ == '__main__':
    test()
