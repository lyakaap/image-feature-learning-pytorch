import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler

import argparse
from tqdm import tqdm
from center_loss import CenterLoss
from models import LeNetPP
from utils import AverageMeter, accuracy, visualize


parser = argparse.ArgumentParser(description='PyTorch Center Loss Example')
parser.add_argument('--dim-hidden', type=int, default=2,
                    help='dimension of hidden layer')
parser.add_argument('--lambda-c', type=float, default=1.0,
                    help='weight parameter of center loss (default: 1.0)')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='learning rate of class center (default: 0.5)')
parser.add_argument('--batch-size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000,
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10,
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--num-workers', type=int, default=2,
                    help='number of workers')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='how frequent to show log (iteration)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()

    model.train()

    ip1_loader = []
    idx_loader = []
    for i, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader)):
        if args.cuda:
            x, y = x.cuda(), y.cuda()
        x, y = Variable(x), Variable(y)

        ip1, pred = model(x)
        loss = criterion[0](pred, y) + criterion[1](y, ip1)

        optimizer[0].zero_grad()
        optimizer[1].zero_grad()

        loss.backward()

        optimizer[0].step()
        optimizer[1].step()

        prec1 = accuracy(pred.data, y.data)
        losses.update(loss.data[0], n=x.size(0))
        top1.update(prec1[0], n=x.size(0))

        ip1_loader.append(ip1)
        idx_loader.append(y)

    feat = torch.cat(ip1_loader, 0)
    labels = torch.cat(idx_loader, 0)
    visualize(feat.data.cpu().numpy(), labels.data.cpu().numpy(), epoch)


def main():
    train_dataset = datasets.MNIST(
        '../data', download=True, train=True, transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ))
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers)

    model = LeNetPP(dim_hidden=args.dim_hidden)
    if args.cuda:
        model = model.cuda()

    nll_loss = nn.NLLLoss()
    center_loss = CenterLoss(dim_hidden=args.dim_hidden, num_classes=10,
                             lambda_c=args.lambda_c, use_cuda=args.cuda)
    if args.cuda:
        nll_loss, center_loss = nll_loss.cuda(), center_loss.cuda()
    criterion = [nll_loss, center_loss]

    optimizer_nn = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = lr_scheduler.StepLR(optimizer_nn, step_size=50, gamma=0.2)

    optimizer_c = optim.SGD(center_loss.parameters(), lr=args.alpha)

    for epoch in range(args.epochs):
        scheduler.step()
        train(train_loader, model, criterion, [optimizer_nn, optimizer_c],
              epoch + 1)


if __name__ == '__main__':
    main()
