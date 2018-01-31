import matplotlib.pyplot as plt


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, top_k=(1,)):
    """Computes the precision@k for the specified values of k"""
    max_k = max(top_k)
    batch_size = target.size(0)

    _, pred = output.topk(max_k, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in top_k:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def visualize(feat, labels, centers, epoch, xy_range=5):
    plt.ion()
    c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff',
         '#ff00ff', '#990000', '#999900', '#009900', '#009999']
    plt.clf()
    for i in range(10):
        plt.plot(feat[labels == i, 0], feat[labels == i, 1], '.', c=c[i])
        plt.text(centers[i, 0], centers[i, 1], str(i), color='black', fontsize=12)
    plt.legend(
        ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    plt.xlim(xmin=-xy_range, xmax=xy_range)
    plt.ylim(ymin=-xy_range, ymax=xy_range)
    plt.text(-xy_range*0.98, xy_range*0.92, 'epoch={}'.format(epoch))
    plt.savefig('../images/epoch-{}.jpg'.format(epoch))
    plt.draw()
    plt.pause(0.001)
