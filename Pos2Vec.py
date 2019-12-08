import chess.pgn
import chess
import pickle

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
    print(games_data[:5])
    print(len(games_data))

    games = [game[0] for game in games_data]
    labels = [game[1] for game in games_data]
    print(games[:5])
    print(labels[:5])
    games_dataset = GamesDataset(games, labels)
