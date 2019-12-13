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
    def __init__(self, parsed_boards, boards, labels, transform=None):
        self.parsed_boards = parsed_boards
        self.boards = boards
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.parsed_boards)

    def __getitem__(self, idx):
        parsed_board = self.parsed_boards[idx]
        board = self.boards[idx]
        outcome = self.labels[idx]

        sample = {'parsed_board': parsed_board, 'board': board, 'outcome': outcome}

        if self.transform:
            sample = self.transform(sample)

        return sample


class BitstringToTensor(object):
    """ Converts a bitstring in sample to Tensors. """

    def __call__(self, sample):
        parsed_board, board, outcome = sample['parsed_board'], sample['board'], sample['outcome']

        # Convert the bitstring to a numpy array:
        # https://stackoverflow.com/questions/29091869/convert-bitstring-string-of-1-and-0s-to-numpy-array
        board_array = np.fromstring(parsed_board,'u1') - ord('0')
        board_tensor = torch.from_numpy(board_array)

        outcome_code = 0
        if (outcome == '1-0'):
            outcome_code = 0
        elif (outcome == '0-1'):
            outcome_code = 1
        else:
            raise Exception('Unexpected outcome.')
        outcome_tensor = torch.tensor(outcome_code)
        #print(outcome, ' -> ', outcome_code, " -> ", outcome_tensor)

        return {'parsed_board': board_tensor, 'board': board, 'outcome': outcome_tensor}

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(773, 600),
            nn.ReLU(True),
            nn.Linear(600, 400),
            nn.ReLU(True),
            nn.Linear(400, 200),
            nn.ReLU(True),
            nn.Linear(200, 100),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(True),
            nn.Linear(200, 400),
            nn.ReLU(True),
            nn.Linear(400, 600),
            nn.ReLU(True),
            nn.Linear(600, 773),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.connect = nn.Linear(773, 400)
        self.lin1 = nn.Linear(400, 200)
        self.lin2 = nn.Linear(200, 100)
        self.lin3 = nn.Linear(100, 2)

    def forward(self, x):
        x = self.connect(x)
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

    # The loss function (which we chose to include as a method of the class, but doesn't need to be)
    # returns the loss and optimizer used by the model
    def get_loss(self, learning_rate):
        # Loss function
        loss = nn.CrossEntropyLoss()
        # Optimizer, self.parameters() returns all the Pytorch operations that are attributes of the class
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        return loss, optimizer

class Combined(nn.Module):
    def __init__(self, modelA, modelB):
        super(Combined, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        self.classifier = nn.Linear(2, 1)
        #self.softmax = nn.Softmax(2)

    def forward(self, x):
        #encoded, decoded = self.modelA(x)
        x = self.modelB(x)
        x = self.classifier(x)
        
        # We need to do x[0] instead of x b/c for some reason,
        # x has shape [1, 1] (as in, it's a nested list like [[0.005]])
        return x[0]
