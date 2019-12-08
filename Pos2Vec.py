import chess.pgn
import chess

import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import save_image

pgn = open('games/2015-05.bare.[6004].pgn')

def boardToBitstring(board):
    """
    Converts the board to a 261 bit representation of the chess board.
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
    return boardBitstring

def pieceToBitstring(piece):
    if piece == None:
        return '0000'

    color = '1' if piece.color else '0'
    type = format(piece.piece_type, '03b')

    return color + type

class BoardDataset(Dataset):
    def __init__(self, boards, labels, transform=None):
        self.boards = boards
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.boards)

    def __getitem__(self, idx):




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



for i in range(0, 10):
    game = chess.pgn.read_game(pgn)

    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        boardBitstring = boardToBitstring(board)
        print(boardBitstring)
