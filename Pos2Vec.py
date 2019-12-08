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


PATH = 'saved_models/autoencoder.pt'

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
        elif (outcome == '0-1'):
            outcome_code = 2
        else:
            raise Exception('Unexpected outcome.')
        outcome_tensor = torch.tensor(outcome_code)
        #print(outcome, ' -> ', outcome_code, " -> ", outcome_tensor)

        return {'board': board_tensor, 'outcome': outcome_tensor}





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
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def trainModel(train_dataloader, test_dataloader):
    #defining some params
    num_epochs = 10 #you can go for more epochs, I am using a mac
    batch_size = 128

    net = Autoencoder()
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in train_dataloader:
            board, outcome = data['board'].float(), data['outcome'].float()
            #print(board)
            #print(outcome)
            #print('\n\n')

            optimizer.zero_grad()
            outputs = net(board)
            #print("outputs:")
            #print(outputs)

            loss = distance(outputs, board)
            loss.backward()
            optimizer.step()
        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for data in test_dataloader:
            board, outcome = data['board'].float(), data['outcome'].float()

            outputs = net(board)
            loss = distance(outputs, board)
            total_test_loss += loss.data.numpy()
        test_loss = total_test_loss / len(test_dataloader)


        print('epoch [{}/{}], train loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.numpy(), test_loss))


    # Save model so it can be loaded:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(net.state_dict(), PATH)



with open('parsed_games/2015-05.bare.[6004].parsed_flattened.pickle', 'rb') as handle:
    games_data = pickle.load(handle)
    #print(games_data[:5])
    #print(len(games_data))

    print("There are", len(games_data), "available for training.")
    training_size = 50000
    games = [game[0] for game in games_data][:training_size]
    outcomes = [game[1] for game in games_data][:training_size]
    #print(games[:2])
    #print(labels[:60])
    print("Running training on", len(games), "games.")


    games_dataset = GamesDataset(boards=games,
                                labels=outcomes,
                                transform=transforms.Compose([BitstringToTensor()]))
    train_size = int(0.75 * len(games_dataset))
    test_size = len(games_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(games_dataset, [train_size, test_size])
    print("Training set size:", train_size)
    print("Testing set size:", test_size)

    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)
    trainModel(train_dataloader, test_dataloader)

    """for i in range(len(games_dataset)):
        sample = games_dataset[i]
        print(i, sample['board'].size(), sample['outcome'])

        if i == 3:
            break
    """
