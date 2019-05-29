import torch
from torch import nn
import torch.nn.functional as F
from doubledummy import DDS
from torch.utils.data import TensorDataset, DataLoader

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(52+52, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 8) 
    
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        retun x

def data_load(path):    
    if os.path.exists('{0}.npz'.format(path.rsplit(".", 1)[0])):
        path = '{0}.npz'.format(path.rsplit(".", 1)[0])
    # csvかnpz(圧縮ファイル)形式
    if path.endswith("csv"):
        data = np.loadtxt(path, delimiter=" ")
        # 次回以降のために圧縮形式で保存
        file_name = "{0}.npz".format(path.rsplit(".", 1)[0])
        np.savez_compressed(file_name, data)
    elif path.endswith("npz"):
        data = np.load(path)["arr_0"]
    else:
        raise NotImplementedError
    data_size = len(data)
    hands, tricks = np.split(data, 208, axis=1)
    return hands, tricks, data_size 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    DDNN = Net()
    player = ['N', 'E', 'S', 'W']
    BATCH_SIZE = 50

    hands, tricks, data_size = data_load(training_path)
    valid = 0.2
    test = 0.2
    
    train_hands, train_tricks = torch.from_numpy(hands[:data_size*(1-valid-test)]), torch.from_numpy(tricks[:data_size*(1-valid-test)])
    valid_hands, valid_tricks = torch.from_numpy(hands[data_size*(1-valid-test):data_size*(1-test)]), torch.from_numpy(tricks[data_size*(1-valid-test):data_size*(1-test)])
    test_hands, test_tricks = torch.from_numpy(hands[data_size*(1-test):]), torch.from_numpy(tricks[data_size*(1-test):])

    train_data = TensorDataset(train_hands, train_tricks)  
    valid_data = TensorDataset(valid_hands, valid_tricks)
    test_data = TensorDataset(test_hands, test_tricks)

    loader_train = DataLoader(train_data, batch_sampler=BATCH_SIZE, shuffle=True)
    loader_valid = DataLoader(valid_data, batch_sampler=1, shuffle=False)
    loader_test = DataLoader(test_data, batch_sampler=1, shuffle=False)
    
    
