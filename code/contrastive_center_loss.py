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
        if self.use_cuda:
            hist = Variable(
                torch.histc(y.cpu().data.float(), bins=self.num_classes, min=0,
                            max=self.num_classes-1) + 1
            ).cuda()
        else:
            hist = Variable(
                torch.histc(y.data.float(), bins=self.num_classes, min=0,
                            max=self.num_classes-1) + 1
            )

        centers_count = hist.index_select(0, y.long())

        # To squeeze the Tensor
        batch_size = hidden.size()[0]
        hidden = hidden.view(batch_size, 1, 1, -1).squeeze()

        centers_pred = self.centers.index_select(0, y.long())
        diff = hidden - centers_pred
        intra_distance = diff.pow(2).sum(1)
        inter_distance = diff.pow(2).sum(1)
        epsilon = 1e-6
        loss = (self.lambda_c / 2.0) * \
               (intra_distance / centers_count).sum() / \
               ((inter_distance / centers_count).sum() + epsilon)
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
    ct = ContrastiveCenterLoss(10, 2)
    y = Variable(torch.LongTensor([0, 0, 0, 1]))
    feat = Variable(torch.zeros(4, 2), requires_grad=True)

    out = ct(y, feat)
    out.backward()


if __name__ == '__main__':
    test()
