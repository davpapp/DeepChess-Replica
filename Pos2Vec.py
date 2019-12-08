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
        print(outcome, ' -> ', outcome_code, " -> ", outcome_tensor)

        return {'board': board_tensor, 'outcome': outcome_tensor}




"""
class Autoencoder(nn.Module):
    def __init__(self):
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True),
            nn.Sigmoid())
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
"""

"""with open('parsed_games/2015-05.bare.[6004].parsed.pickle', 'rb') as handle:
    game_data = pickle.load(handle)
    print(game_data[0])
    print(len(game_data))
"""


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

    for i in range(len(games_dataset)):
        sample = games_dataset[i]
        print(i, sample['board'].size(), sample['outcome'].size())

        if i == 100:
            break
