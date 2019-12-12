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

#import Train


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
        # Uncomment the line below if training
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
        x = self.classifier(x)
        return x


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

def boardToBitstring(board):
    """
    Converts the board to a 773 bit representation of the chess board.
    I'm not sure pieceToBitstring is correct.
    """
    boardBitstring = ''
    for i in range(0, 8):
        for j in range(0, 8):
            piece = board.piece_at(chess.square(i, j))
            pieceBitstring = pieceToBitstring(piece)
            boardBitstring += pieceBitstring

    sideToMove = '1' if board.turn == chess.WHITE else '0'
    whiteCastlingRight = '1' if board.has_kingside_castling_rights(chess.WHITE) else '0'
    whiteCastlingRightQueenside = '1' if board.has_queenside_castling_rights(chess.WHITE) else '0'
    blackCastlingRight = '1' if board.has_kingside_castling_rights(chess.BLACK) else '0'
    blackCastlingRIghtQueenside = '1' if board.has_queenside_castling_rights(chess.BLACK) else '0'

    boardBitstring += sideToMove + whiteCastlingRight + whiteCastlingRightQueenside + blackCastlingRight + blackCastlingRIghtQueenside
    #print(len(boardBitstring))
    return boardBitstring

def pieceToBitstring(piece):
    if piece == None:
        return '000000000000'

    is_white_pawn = '1' if piece.piece_type == chess.PAWN and piece.color else '0'
    is_white_knight = '1' if piece.piece_type == chess.KNIGHT and piece.color else '0'
    is_white_bishop = '1' if piece.piece_type == chess.BISHOP and piece.color else '0'
    is_white_rook = '1' if piece.piece_type == chess.ROOK and piece.color else '0'
    is_white_queen = '1' if piece.piece_type == chess.QUEEN and piece.color else '0'
    is_white_king = '1' if piece.piece_type == chess.KING and piece.color else '0'

    is_black_pawn = '1' if piece.piece_type == chess.PAWN and not piece.color else '0'
    is_black_knight = '1' if piece.piece_type == chess.KNIGHT and not piece.color else '0'
    is_black_bishop = '1' if piece.piece_type == chess.BISHOP and not piece.color else '0'
    is_black_rook = '1' if piece.piece_type == chess.ROOK and not piece.color else '0'
    is_black_queen = '1' if piece.piece_type == chess.QUEEN and not piece.color else '0'
    is_black_king = '1' if piece.piece_type == chess.KING and not piece.color else '0'

    return is_white_pawn + is_white_knight + is_white_bishop + is_white_rook + is_white_queen + is_white_king + is_black_pawn + is_black_knight + is_black_bishop + is_black_rook + is_black_queen + is_black_king


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

