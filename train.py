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
        #x = self.decoder(x)
        return x

class Evaluator(nn.Module):
    def __init__(self):
        super(Evaluator, self).__init__()
        self.connect = nn.Linear(100, 400)
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

    def forward(self, x):
        x = self.modelA(x)
        x = self.modelB(x)
        return x


def trainModel(train_dataloader, test_dataloader):
    #defining some params
    num_epochs = 10 #you can go for more epochs, I am using a mac
    batch_size = 128

    print('Training model...')

    autoenc = Autoencoder()
    #    Load state dicts
    autoenc.load_state_dict(torch.load('saved_models/autoencoder_26.pt'))

    evaluate = Evaluator()

    net = Combined(autoenc, evaluate)

    #net.load_state_dict(torch.load('saved_models/autoencoder.pt'))
    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-5)

    for epoch in range(num_epochs):
        print(epoch)
        for data in train_dataloader:
            board, outcome = data['board'].float(), data['outcome'].float()
            #print(board)
            #print(outcome)
            #print('\n\n')

            optimizer.zero_grad()
            outputs = net(board)
            #print("outputs:")
            #print(outputs)

            loss = distance(outputs, outcome)
            loss.backward()
            optimizer.step()
        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for data in test_dataloader:
            board, outcome = data['board'].float(), data['outcome'].float()

            outputs = net(board)
            loss = distance(outputs, outcome)
            total_test_loss += loss.data.numpy()
        test_loss = total_test_loss / len(test_dataloader)


        print('epoch [{}/{}], train loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.numpy(), test_loss))


    # Save model so it can be loaded:
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    path = 'saved_models/board_classifier.pt'
    torch.save(net.state_dict(), path)
    #torch.save(net.state_dict(), PATH)

def validateModel():
    # load board_classifier
    # Define some boards
    # convert boards to Dataset then dataloader
    # run model on dataloader and show results


with open('parsed_games/2015-05.bare.[6004].parsed_flattened.pickle', 'rb') as handle:
    games_data = pickle.load(handle)
    #print(games_data[:5])
    #print(len(games_data))

    print("There are", len(games_data), "available for training.")
    training_size = 1000
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

    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    print('created train dataloader')
    test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True)
    print('created test dataloader')
    trainModel(train_dataloader, test_dataloader)
    validateModel()
