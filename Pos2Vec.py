import chess.pgn
import chess
import pickle
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader



class GamesDataset(Dataset):
    def __init__(self, boards, labels, transform=None):
        self.boards = boards
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):
        board = self.boards[idx]
        outcome = self.labels[idx]

        sample = {'board': board, 'outcome': outcome}
        if self.transform:
            sample = self.transform(sample)

        return sample


class BitstringToTensor(object):
    """ Converts a bitstring in sample to Tensors. """

    def __call__(self, sample):
        board, outcome = sample['board'], sample['outcome']

        # Convert the bitstring to a numpy array:
        # https://stackoverflow.com/questions/29091869/convert-bitstring-string-of-1-and-0s-to-numpy-array
        board_array = np.fromstring(board,'u1') - ord('0')
        board_tensor = torch.from_numpy(board_array)

        outcome_code = 0
        if (outcome == '1-0'):
            outcome_code = 0
        elif (outcome == '1/2-1/2'):
            outcome_code = 1
        else:
            outcome_code = 2
        outcome_tensor = torch.tensor(outcome_code)
        #print(outcome, ' -> ', outcome_code, " -> ", outcome_tensor)

        return {'board': board_tensor, 'outcome': outcome_tensor}





class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(761, 600),
            nn.ReLU(True),
            nn.Linear(600, 400),
            nn.ReLU(True),
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Linear(400, 600),
            nn.ReLU(True),
            nn.Linear(600, 761),
            nn.ReLU(True),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def trainModel(dataloader):
    #defining some params
    num_epochs = 5 #you can go for more epochs, I am using a mac
    batch_size = 128

    model = Autoencoder().cpu()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(),weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in dataloader:
            print(data)
            board, outcome = data
            board = Variable(board).cpu()
            # ===================forward=====================
            output = model(outcome)
            loss = distance(output, img)
            # ===================backward====================
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch+1, num_epochs, loss.data()))



with open('parsed_games/2015-05.bare.[6004].parsed_flattened.pickle', 'rb') as handle:
    games_data = pickle.load(handle)
    #print(games_data[:5])
    #print(len(games_data))

    games = [game[0] for game in games_data]
    outcomes = [game[1] for game in games_data]
    #print(games[:2])
    #print(labels[:60])

    games_dataset = GamesDataset(boards=games,
                                labels=outcomes,
                                transform=transforms.Compose([BitstringToTensor()]))

    dataloader = DataLoader(games_dataset, batch_size=4, shuffle=True)
    trainModel(dataloader)

    """for i in range(len(games_dataset)):
        sample = games_dataset[i]
        print(i, sample['board'].size(), sample['outcome'])

        if i == 5:
            break"""
