from __future__ import print_function
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from utils import plain_log
from npn import NPNLinear
from npn import NPNSigmoid
from npn import NPNRelu
from npn import NPNDropout
from npn import multi_logistic_loss
from npn import NPNBCELoss
from npn import KL_BG
from npn import KL_loss
from npn import RMSE
from datasets_boston_housing import Dataset_boston_housing
from torch.utils import data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_workers', type=int, default=2,
                    help='number of workers')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--output_s', type=float, default=1.0,
                    help='lambda of output_s')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--evaluate', action='store_true', default=False,
                    help='evaluate only')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--checkpoint', type=str, default='none',
                    help='file name of checkpoint model')
parser.add_argument('--save_interval', type=int, default=100, metavar='N',
                    help='how many epochs to wait before saving model')
parser.add_argument('--log_file', type=str, default='tmp',
                    help='log file name')
parser.add_argument('--save_head', type=str, default='tmp',
                    help='file name head for saving')
parser.add_argument('--type', type=str, default='mlp',
                    help='mlp/npn')
parser.add_argument('--loss', type=str, default='default',
                    help='default/npnbce/kl')
parser.add_argument('--num_train', type=int, default=60000,
                    help='num train')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
random.seed(args.seed)
torch.manual_seed(int(args.seed))

if args.type.startswith('regress_'):
    bh_train_dataset = Dataset_boston_housing('./boston_housing_nor_train.pkl')
    bh_val_dataset = Dataset_boston_housing('./boston_housing_nor_val.pkl')

    train_loader = data.DataLoader(
        dataset = bh_train_dataset,
        batch_size = args.batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = False
    )

    test_loader = data.DataLoader(
        dataset = bh_val_dataset,
        batch_size = args.test_batch_size,
        shuffle = False,
        num_workers = args.num_workers,
        pin_memory = False
    )
else:
    mnist_train = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ]))
    size_train = len(mnist_train)
    indices = list(range(size_train))
    np.random.shuffle(indices)
    num_train = args.num_train
    train_ind = indices[:num_train]
    train_sampler = SubsetRandomSampler(train_ind)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.batch_size, sampler=train_sampler, **kwargs)
        #batch_size=args.batch_size, shuffle=False, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.Compose([
                           transforms.ToTensor()
                       ])),
        batch_size=args.test_batch_size, shuffle=False, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)
        self.dropout = args.dropout

        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = nn.Dropout(self.dropout)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.sigmoid(self.fc1(x))
        x = self.drop1(x)
        x = F.sigmoid(self.fc2(x))
        x = self.drop2(x)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x)
        ##x = torch.log(F.sigmoid(x))
        #x = torch.log(F.softmax(F.sigmoid(x)))
        #return x

class NPNNet(nn.Module):
    def __init__(self):
        super(NPNNet, self).__init__()
        self.dropout = args.dropout

        self.fc1 = NPNLinear(784, 800, False)
        self.sigmoid1 = NPNSigmoid()
        #self.sigmoid1 = NPNRelu()
        self.fc2 = NPNLinear(800, 800)
        self.sigmoid2 = NPNSigmoid()
        #self.sigmoid2 = NPNRelu()
        self.fc3 = NPNLinear(800, 10)
        self.sigmoid3 = NPNSigmoid()

        self.drop1 = NPNDropout(self.dropout)
        self.drop2 = NPNDropout(self.dropout)
        self.drop3 = NPNDropout(self.dropout)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.sigmoid1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.sigmoid2(x)
        x = self.drop2(x)
        x, s = self.sigmoid3(self.fc3(x))
        return x, s

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.dropout = args.dropout

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.drop1 = nn.Dropout(self.dropout)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        #x = F.dropout(x, training=self.training)
        x = self.drop1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

class NPNCNN(nn.Module):
    def __init__(self):
        super(NPNCNN, self).__init__()
        self.dropout = args.dropout
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = NPNLinear(320, 50, dual_input=False)
        self.relu1 = NPNRelu()
        self.drop1 = NPNDropout(self.dropout)
        self.fc2 = NPNLinear(50, 10)
        self.sigmoid1 = NPNSigmoid()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = self.relu1(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        if args.loss == 'nll':
            x, _ = x
            return F.log_softmax(x), x
        else:
            x, s = self.sigmoid1(x)
            return x, s

class ReNPN(nn.Module):
    def __init__(self):
        super(ReNPN, self).__init__()
        self.dropout = args.dropout

        self.fc1 = NPNLinear(13, 50, False)
        self.relu1 = NPNRelu()
        self.fc2 = NPNLinear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x, s = self.fc2(x)
        return x, s

class ReMLP(nn.Module):
    def __init__(self):
        super(ReMLP, self).__init__()
        self.dropout = args.dropout

        self.fc1 = nn.Linear(13, 50)
        self.fc2 = nn.Linear(50, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

if args.type == 'mlp':
    model = Net()
elif args.type == 'npn':
    model = NPNNet()
elif args.type == 'cnn':
    model = CNN()
elif args.type == 'npncnn':
    model = NPNCNN()
elif args.type == 'regress_npn':
    model = ReNPN()
elif args.type == 'regress_mlp':
    model = ReMLP()
if args.cuda:
    model.cuda()

#optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
optimizer = optim.Adadelta(model.parameters(), lr = args.lr, eps = 1e-7) # lr default 0.02
#optimizer = optim.Adam(model.parameters(), lr = args.lr) # lr 
ind = list(range(args.batch_size))
ind_test = list(range(1000))
bce = nn.BCELoss()
mse = nn.MSELoss()

def train(epoch):
    model.train()
    sum_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        # TODO: expand label here
        if not args.type.startswith('regress_'):
            target_ex = torch.zeros(target.size()[0], 10)
            target_ex[ind[:min(args.batch_size, target.size()[0])], target] = 1

        if args.type == 'npn' or args.type == 'npncnn':
            target = target_ex
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        if args.type == 'mlp' or args.type == 'cnn':
            loss = F.nll_loss(output, target)
        else:
            if args.type != 'regress_mlp':
                x, s = output
            #loss = F.nll_loss(torch.log(x+1e-10), target) + args.output_s * torch.sum(s)
            #loss = multi_logistic_loss(x, target) + args.output_s * torch.sum(s)
            if args.loss == 'default':
                loss = bce(x, target) + args.output_s * torch.sum(s ** 2)
            elif args.loss == 'npnbce':
                loss = NPNBCELoss(x, s, target) + args.output_s * torch.sum(s ** 2)
            elif args.loss == 'kl':
                loss = KL_BG(x, s, target) + args.output_s * torch.sum(s ** 2)
            elif args.loss == 'nll':
                loss = F.nll_loss(x, target)
            elif args.loss == 'gaussian':
                loss = KL_loss(output, target) + 0.5 * args.output_s * torch.sum(s ** 2)
            elif args.loss == 'mse':
                loss = mse(output, target)
            # TODO: use BCELoss
        sum_loss += loss.data[0]
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0 and batch_idx != 0:
            log_txt = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.7f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0])
            print(log_txt)
            plain_log(args.log_file,log_txt+'\n')
    avg_loss = sum_loss / len(train_loader.dataset) * args.batch_size
    log_txt = 'Train Epoch {}: Average Loss = {:.7f}'.format(epoch, avg_loss)
    print(log_txt)
    plain_log(args.log_file,log_txt+'\n')
    if epoch % args.save_interval == 0 and epoch != 0:
        torch.save(model, '%s.model' % args.save_head)

def test():
    model.eval()
    test_loss = 0
    rmse_loss = 0
    correct = 0
    for data, target in test_loader:
        if not args.type.startswith('regress_'):
            target_ex = torch.zeros(target.size()[0], 10)
            target_ex[ind_test[:min(1000, target.size()[0])], target] = 1

        if args.cuda:
            if not args.type.startswith('regress_'):
                data, target_ex, target = data.cuda(), target_ex.cuda(), target.cuda()
            else:
                data, target = data.cuda(), target.cuda()
        if not args.type.startswith('regress_'):
            data, target_ex = Variable(data, volatile=True), Variable(target_ex)
        else:
            data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        if args.type == 'npn' or args.type == 'npncnn' or args.type.startswith('regress_'):
            if args.type != 'regress_mlp':
                output, s = output
            #test_loss += F.nll_loss(torch.log(output+1e-10), target, size_average=False).data[0] # sum up batch loss
            if args.loss == 'default':
                test_loss += (bce(output, target_ex) + args.output_s * torch.sum(s ** 2)).data[0]
            elif args.loss == 'npnbce':
                test_loss += (NPNBCELoss(output, s, target_ex) + args.output_s * torch.sum(s ** 2)).data[0]
            elif args.loss == 'kl':
                test_loss += (KL_BG(output, s, target_ex) + args.output_s * torch.sum(s ** 2)).data[0]
            elif args.loss == 'gaussian':
                test_loss += KL_loss((output, s), target).data[0]
                rmse_loss += RMSE(output, target).data[0]
            elif args.loss == 'mse':
                test_loss += mse(output, target).data[0]
                rmse_loss += RMSE(output, target).data[0]
        else:
            test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        if not args.type.startswith('regress_'):
            pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    if not args.type.startswith('regress_'):
        log_txt = 'Test set: Average loss: {:.7f}, Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset))
    else:
        log_txt = 'Test set: Average loss: {:.7f}, RMSE: {:.6f}'.format(test_loss, rmse_loss)
    print(log_txt)
    plain_log(args.log_file,log_txt+'\n')

if args.checkpoint != 'none':
    model = torch.load(args.checkpoint)
    print(str(model))
    for key, module in model._modules.items():
        print('key', key)
        print('module', module)
        if module.__class__.__name__ == 'NPNLinear':
            print('para\n', torch.log(torch.exp(module.W_s_[:8,:8])+1))

if not args.evaluate:
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        if epoch % 1 == 0:
            test()
test()
