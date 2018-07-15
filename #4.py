# coding: utf-8

# In[19]:


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils_3 import mnist, plot_graphs


# In[20]:


train_loader, valid_loader, test_loader = mnist(valid=10000)


# In[88]:


class Net(nn.Module):
    def __init__(self, lr=1e-4, l2=0.):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        nn.init.xavier_uniform_(self.fc1.weight)
        self.fc2 = nn.Linear(256, 256)  #
        nn.init.xavier_uniform_(self.fc2.weight)
        self.fc3 = nn.Linear(256, 128)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc3.weight)
        self.fc4 = nn.Linear(128, 64)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc4.weight)
        self.fc5 = nn.Linear(64, 32)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc5.weight)
        self.fc6 = nn.Linear(32, 16)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc6.weight)
        self.fc7 = nn.Linear(16, 8)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc7.weight)
        self.fc8 = nn.Linear(8, 4)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc8.weight)
        self.fc9 = nn.Linear(4, 2)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc9.weight)
        self.fc10 = nn.Linear(2, 2)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc10.weight)
        self.fc11 = nn.Linear(2, 2)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc11.weight)
        self.fc12 = nn.Linear(2, 4)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc12.weight)
        self.fc13 = nn.Linear(4, 8)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc13.weight)
        self.fc14 = nn.Linear(8, 16)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc14.weight)
        self.fc15 = nn.Linear(16, 32)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc15.weight)
        self.fc16 = nn.Linear(32, 64)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc16.weight)
        self.fc17 = nn.Linear(64, 128)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc17.weight)
        self.fc18 = nn.Linear(128, 256)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc18.weight)
        self.fc19 = nn.Linear(256, 256)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc19.weight)
        self.fc20 = nn.Linear(256, 256)  # nn.Linear(128, 256)
        nn.init.xavier_uniform_(self.fc20.weight)
        self.bn = nn.BatchNorm1d(256)
        self.optim = optim.Adam(self.parameters(), lr=lr, weight_decay=l2)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.bn(x)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc3(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc4(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc5(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc6(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc7(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc8(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc9(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc10(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc11(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc12(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc13(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc14(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc15(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc16(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc17(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc18(x))
        x = F.dropout(x, 0.5)
        x = F.relu(self.fc19(x))
        x = F.dropout(x, 0.5)

        x = self.fc20(x)
        x = F.log_softmax(x, dim=1)
        return x

    def loss(self, output, target, **kwargs):
        self._loss = F.nll_loss(output, target, **kwargs)
        self._correct = output.data.max(1, keepdim=True)[1]
        self._correct = self._correct.eq(target.data.view_as(self._correct)).to(torch.float).cpu().mean()
        return self._loss


# In[89]:


models = {'default': Net()}
train_log = {k: [] for k in models}
valid_log = {k: [] for k in models}
test_log = {k: [] for k in models}


# In[7]:


def train(epoch, models, log=None):
    train_size = len(train_loader.sampler)
    for batch_idx, (data, target) in enumerate(train_loader):
        for model in models.values():
            model.optim.zero_grad()
            output = model(data)
            loss = model.loss(output, target)
            loss.backward()
            model.optim.step()

        if batch_idx % 200 == 0:
            line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
                epoch, batch_idx * len(data), train_size, 100. * batch_idx / len(train_loader))
            losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
            print(line + losses)

    else:
        batch_idx += 1
        line = 'Train Epoch: {} [{}/{} ({:.0f}%)]\tLosses '.format(
            epoch, batch_idx * len(data), train_size, 100. * batch_idx / len(train_loader))
        losses = ' '.join(['{}: {:.6f}'.format(k, m._loss.item()) for k, m in models.items()])
        if log is not None:
            for k in models:
                log[k].append((models[k]._loss, models[k]._correct))
        print(line + losses)


# In[8]:


def test(models, loader, log=None):
    test_size = len(loader.sampler)
    avg_lambda = lambda l: 'Loss: {:.4f}'.format(l)
    acc_lambda = lambda c, p: 'Accuracy: {}/{} ({:.0f}%)'.format(c, test_size, p)
    line = lambda i, l, c, p: '{}: '.format(i) + avg_lambda(l) + '\t' + acc_lambda(c, p)

    test_loss = {k: 0. for k in models}
    correct = {k: 0. for k in models}
    with torch.no_grad():
        for data, target in loader:
            output = {k: m(data) for k, m in models.items()}
            for k, m in models.items():
                test_loss[k] += m.loss(output[k], target, size_average=False).item()  # sum up batch loss
                pred = output[k].data.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct[k] += pred.eq(target.data.view_as(pred)).cpu().sum()

    for k in models:
        test_loss[k] /= test_size
    correct_pct = {k: c.to(torch.float) / test_size for k, c in correct.items()}
    lines = '\n'.join([line(k, test_loss[k], correct[k], 100 * correct_pct[k]) for k in models]) + '\n'
    report = 'Test set:\n' + lines
    if log is not None:
        for k in models:
            log[k].append((test_loss[k], correct_pct[k]))
    print(report)


# In[9]:


if __name__ == '__main__':
    for epoch in range(1, 101):
        for model in models.values():
            model.train()
        train(epoch, models, train_log)
        for model in models.values():
            model.eval()
        test(models, valid_loader, valid_log)
        test(models, test_loader, test_log)
    plot_graphs(test_log, 'loss')
    plot_graphs(test_log, 'accuracy')

