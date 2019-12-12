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

from Train import Autoencoder
from Train import Evaluator
from Train import Combined

from GameParser import boardToBitstring
from GameParser import pieceToBitstring


DEEPCHESS_PATH = 'saved_models/deepchess/'


def find_best_move(board, white):
    best_score = -1
    best_move = ""
    for move in board.legal_moves:
        #print(move)
        #print(board)
        board.push(move)
        board_score = eval_board(board, white)
        if board_score > best_score:
            best_score = board_score
            best_move = move
        board.pop()

    return best_move

def eval_board(board, white):

    bit_board = boardToBitstring(board)

    board_array = np.fromstring(bit_board,'u1') - ord('0')
    board_tensor = torch.from_numpy(board_array).float()

    #bit_board = [float(b) for b in bit_board]
    #bit_board = np.asarray(bit_board)
    ##tensor_board = torch.from_numpy(bit_board)
    #tensor_board = torchvision.transforms.Compose([BitstringToTensor(tensor_board)])#tensor_board.type(torch.DoubleTensor) #tensor_board


    #.transforms.Compose([BitstringToTensor()])
    #tensor_board = torch.tensor(bit_board)
    autoencoder = Autoencoder()
    #autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH + 'autoencoder_7.pt'))

    evaluator = Evaluator()
    #autoencoder.load_state_dict(torch.load(AUTOENCODER_PATH + 'autoencoder_7.pt'))

    net = Combined(autoencoder, evaluator)
    net.load_state_dict(torch.load(DEEPCHESS_PATH + 'board_classifier.pt'))
    #net = net.float()


    output = net(board_tensor)
    return output if white else -output


board = chess.Board()

white = True
for i in range(10):
    move = find_best_move(board, white)
    print(move)
    board.push(move)
    white = not white
    print(board)
