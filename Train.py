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
from torch.utils.tensorboard import SummaryWriter

from Models import GamesDataset
from Models import BitstringToTensor
from Models import Autoencoder
from Models import Evaluator
from Models import Combined

print("Cuda available: ", torch.cuda.is_available())
if torch.cuda.is_available():
  dev = "cuda:0"
else:
  dev = "cpu"
device = torch.device(dev)
print("Using device:", device, "\n")
#import pycuda.driver as cuda
#cuda.init()
## Get Id of default device
#torch.cuda.current_device()
# 0
#cuda.Device(0).name() # '0' is the id of your GPU
# Tesla K80

AUTOENCODER_PATH = 'saved_models/autoencoder/'
DEEPCHESS_PATH = 'saved_models/deepchess/'

writer = SummaryWriter()


def trainAutoencoder(train_dataloader, test_dataloader):
    num_epochs = 100
    batch_size = 128

    last_epoch = 0
    autoencoder = Autoencoder().to(device)

    if (last_epoch > 0):
        print("Continuing training of autoencoder_" + last_epoch + ".pt...")
        autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH + 'autoencoder_' + str(last_epoch) + '.pt'))
    else:
        print("Training from scratch...")

    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(),weight_decay=1e-5)


    for epoch in range(last_epoch + 1, num_epochs + last_epoch):
        for data in train_dataloader:
            parsed_board, outcome = data['parsed_board'].float().to(device), data['outcome'].float().to(device)

            optimizer.zero_grad()
            encoded, decoded = autoencoder(parsed_board)
            loss = distance(parsed_board, decoded)
            loss.backward()
            optimizer.step()

        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for data in test_dataloader:
            parsed_board, outcome = data['parsed_board'].float().to(device), data['outcome'].float().to(device)
            encoded, decoded = autoencoder(parsed_board)
            loss = distance(parsed_board, decoded)
            total_test_loss += loss.data.numpy()

        test_loss = total_test_loss / len(test_dataloader)
        writer.add_scalar('training loss', loss.data.numpy(), epoch)
        writer.add_scalar('test loss', test_loss, epoch)

        torch.save(autoencoder.state_dict(), AUTOENCODER_PATH + 'autoencoder_' + str(epoch) + '.pt')

        print('epoch [{}/{}], train loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.numpy(), test_loss))

    # Write to TensorBoard for visualization purposes
    dataiter = iter(train_dataloader)
    data = dataiter.next()
    parsed_board, outcome = data['parsed_board'].float(), data['outcome'].float()
    writer.add_graph(autoencoder, parsed_board)
    writer.close()


def trainDeepChess(train_dataloader, test_dataloader):
    num_epochs = 2
    batch_size = 128

    autoencoder = Autoencoder()
    autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH + 'autoencoder_23.pt'))

    evaluator = Evaluator()

    net = Combined(autoencoder, evaluator).to(device)

    distance = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(),weight_decay=1e-5)

    for epoch in range(num_epochs):
        for data in train_dataloader:
            parsed_board, outcome = data['parsed_board'].float().to(device), data['outcome'].float().to(device)

            optimizer.zero_grad()
            outputs = net(parsed_board)

            loss = distance(outputs, outcome)
            loss.backward()
            optimizer.step()
        # At the end of the epoch, do a pass on the test set
        total_test_loss = 0
        for data in test_dataloader:
            parsed_board, outcome = data['parsed_board'].float().to(device), data['outcome'].float().to(device)

            outputs = net(parsed_board)

            """print(data['board'])
            print("Actual outcome:", outcome)
            print("Predicted outcome:", outputs)
            print('\n')"""

            loss = distance(outputs, outcome)
            total_test_loss += loss.data.cpu().numpy()
        test_loss = total_test_loss / len(test_dataloader)

        torch.save(autoencoder.state_dict(), DEEPCHESS_PATH + 'deepchess_' + str(epoch) + '.pt')

        print('epoch [{}/{}], train loss:{:.4f}, test_loss:{:.4f}'.format(epoch+1, num_epochs, loss.data.cpu().numpy(), test_loss))

    # Write to TensorBoard for visualization purposes
    dataiter = iter(train_dataloader)
    data = dataiter.next()
    parsed_board, outcome = data['parsed_board'].float().to(device), data['outcome'].float().to(device)
    writer.add_graph(net, parsed_board)
    #torch.save(net.state_dict(), DEEPCHESS_PATH + 'board_classifier.pt')
    writer.close()

    validateDeepChess(net, test_dataloader)

def validateDeepChess(net, test_dataloader):
    """
    Pick a random board and run DeepChess on it to help visualize results.
    """
    pass
    idx = 0
    for data in test_dataloader:
        if idx > 5:
            break

        parsed_board, board_fen, outcome = data['parsed_board'].float(), data['board'], data['outcome'].float()
        # reconstruct board
        print('\n\n')
        print("FEN representation of board:", board_fen)
        board = chess.Board(fen=board_fen[0])
        print(board)

        output = net(parsed_board)
        print("Probability of White winning:", output, ". Actual outcome:", outcome)
        idx += 1



with open('parsed_games/2015-05.bare.[6004].parsed_flattened.pickle', 'rb') as handle:
    games_data = pickle.load(handle)

    print("There are", len(games_data), "available for training.")
    training_size = 120000
    parsed_boards = [game[0] for game in games_data][:training_size]
    boards = [game[1] for game in games_data][:training_size]
    outcomes = [game[2] for game in games_data][:training_size]
    #print(games[:2])
    #print(labels[:60])
    print("Running training on", len(parsed_boards), "games.")


    games_dataset = GamesDataset(parsed_boards=parsed_boards,
                                boards=boards,
                                labels=outcomes,
                                transform=transforms.Compose([BitstringToTensor()]))
    train_size = int(0.9 * len(games_dataset))
    test_size = len(games_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(games_dataset, [train_size, test_size])
    print("Training set size:", train_size)
    print("Testing set size:", test_size)

    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True)

    #what_to_train = 'AUTOENCODER'
    what_to_train = 'DEEPCHESS'
    if what_to_train == 'AUTOENCODER':
        print("Training Autoencoder...\n\n")
        trainAutoencoder(train_dataloader, test_dataloader)
    elif what_to_train == 'DEEPCHESS':
        print("Training DeepChess...\n\n")
        trainDeepChess(train_dataloader, test_dataloader)
