import torch
import torch.nn as nn
from torch.autograd import Variable


class CenterLoss(nn.Module):
    def __init__(self, dim_hidden, num_classes, lambda_c=1.0, use_cuda=True):
        super(CenterLoss, self).__init__()
        self.dim_hidden = dim_hidden
        self.num_classes = num_classes
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, dim_hidden))
        self.use_cuda = use_cuda

    def forward(self, y, hidden):
        batch_size = hidden.size()[0]

        expanded_centers = self.centers.index_select(dim=0, index=y.long())
        intra_distance = hidden - expanded_centers
        loss = (self.lambda_c / 2.0) * intra_distance.pow(2).sum() / batch_size
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
    ct = CenterLoss(2, 2, use_cuda=False)
    y = Variable(torch.LongTensor([0, 0, 0, 1]))
    feat = Variable(torch.zeros(4, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()


if __name__ == '__main__':
    test()
