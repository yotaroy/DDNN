import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(52+52, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 5) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

class Data:
    def __init__(self, train_path, test_path, valid_rate=0.25):
        self.train_path = train_path
        self.test_path = test_path
        self.valid_rate = valid_rate
        self.train_size = -1
        self.test_size = -1
    
    def data_load(self, path):    
        if os.path.exists('{0}.npz'.format(path.rsplit(".", 1)[0])):
            path = '{0}.npz'.format(path.rsplit(".", 1)[0])
        # csvかnpz(圧縮ファイル)形式
        if path.endswith("csv"):
            data = np.loadtxt(path)
            # 次回以降のために圧縮形式で保存
            file_name = "{0}.npz".format(path.rsplit(".", 1)[0])
            np.savez_compressed(file_name, data)
        elif path.endswith("npz"):
            data = np.load(path)["arr_0"]
        else:
            raise NotImplementedError
        hands, tricks = np.split(data, [208], axis=1)
        return hands, tricks
    
    def train_data(self):
        hands, tricks = self.data_load(self.train_path)
        if self.train_size == -1:
            self.train_size = len(hands)
        hands = hands[:int(self.train_size*(1-self.valid_rate))]
        tricks = tricks[:int(self.train_size*(1-self.valid_rate))]
        return torch.from_numpy(hands), torch.from_numpy(tricks)

    def valid_data(self):
        hands, tricks = self.data_load(self.train_path)
        if self.train_size == -1:
            self.train_size = len(hands)
        hands = hands[int(self.train_size*(1-self.valid_rate)):]
        tricks = tricks[int(self.train_size*(1-self.valid_rate)):]
        return torch.from_numpy(hands), torch.from_numpy(tricks)
    
    def test_data(self):
        hands, tricks =  self.data_load(self.test_path)
        if self.test_size == -1:
            self.test_size = len(hands)
        return torch.from_numpy(hands), torch.from_numpy(tricks)
    
def train(epoch, batch_size):
    train_loss = 0
    for i in range(4):
        for hand, trick in train_loader:
            p_n, p_e, p_s, p_w = torch.chunk(hand, 4, dim=1)
            player = [p_n, p_e, p_s, p_w]
            n, e, s, w = torch.chunk(trick, 4, dim=1)
            result = [n, e, s, w]
            
            declarer = player[i].to(device)
            partner = player[(i+2)%4].to(device)
            dds = torch.div(torch.add(result[i%2], result[(i+2)%2]), 2).float().to(device)

            optimizer.zero_grad()
            output = model(torch.cat((declarer, partner), dim=1).float().to(device))
            loss = F.mse_loss(output, dds)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
    train_loss /= (len(train_loader.dataset) * 4 / batch_size)

    print('epoch:{} train_loss={}'.format(epoch, train_loss))
    return train_loss


def test(loader, batch_size, name='test'):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        for hand, trick in loader:
            declarer, _, partner, _ = torch.chunk(hand, 4, dim=1)
            n, e, s, w = torch.chunk(trick, 4, dim=1)
            result = [n, e, s, w]
            
            dds = torch.div(torch.add(result[0], result[2]), 2).float().to(device)

            output = model(torch.cat((declarer, partner), dim=1).float().to(device))
            test_loss += F.mse_loss(output, dds).item()

        test_loss /= (len(loader.dataset) / batch_size)

        print('{}_loss={}'.format(name, test_loss))
        return test_loss

def graph_plot(path):
    x = range(len(train_loss))
    plt.plot(x, train_loss, label='train')
    plt.plot(x, valid_loss, label='validation')
    plt.yscale('log')
    plt.legend()
    plt.savefig(path)

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    BATCH_SIZE = 50

    data = Data(train_path='dataset100000_5.csv', test_path='dataset1000.csv')
    
    train_hands, train_tricks = data.train_data()
    valid_hands, valid_tricks = data.valid_data()
    test_hands, test_tricks = data.test_data()

    train_data = TensorDataset(train_hands, train_tricks)  
    valid_data = TensorDataset(valid_hands, valid_tricks)
    test_data = TensorDataset(test_hands, test_tricks)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    model = Network().to(device)
    optimizer = optim.Adam(model.parameters())    

    EPOCH = 10 ** 5

    train_loss = [np.nan]
    valid_loss = [] 

    start = time.time()
    valid_loss.append(test(valid_loader, BATCH_SIZE, 'valid'))
    for e in range(EPOCH):
        print('--------------------')
        train_loss.append(train(e, BATCH_SIZE))
        valid_loss.append(test(valid_loader, BATCH_SIZE, 'valid'))
        if (e+1) % 1000 == 0:
            graph_plot('ex_dataset5.png')
    print('total time =', time.time() - start)
        