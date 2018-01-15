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
        # torch.histc is not implemented on CPU
        # To calculate the total number of every class in one mini-batch.
        # See Equation 4 in the paper
        if self.use_cuda:
            hist = Variable(
                torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0,
                            max=self.num_classes) + 1
            ).cuda()
        else:
            hist = Variable(
                torch.histc(y.data.float(), bins=self.num_classes, min=0,
                            max=self.num_classes) + 1
            )

        centers_count = hist.index_select(0, y.long())

        # To squeeze the Tensor
        batch_size = hidden.size()[0]
        hidden = hidden.view(batch_size, 1, 1, -1).squeeze()

        # To check the dim of centers and features
        if hidden.size()[1] != self.dim_hidden:
            raise ValueError(
                "Center's dim: {0} "
                "should be equal to input feature's dim: {1}".format(
                    self.dim_hidden, hidden.size()[1]))

        centers_pred = self.centers.index_select(0, y.long())
        diff = hidden - centers_pred
        loss = (self.lambda_c / 2.0) * (diff.pow(2).sum(1) / centers_count).sum()
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
    ct = CenterLoss(10, 2)
    y = Variable(torch.LongTensor([0, 0, 0, 1]))
    feat = Variable(torch.zeros(4, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()


if __name__ == '__main__':
    test()
