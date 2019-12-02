import chess.pgn
import chess

pgn = open('games/2015-05.bare.[6004].pgn')

first_game = chess.pgn.read_game(pgn)
second_game = chess.pgn.read_game(pgn)

print(first_game.headers['Result'])


def boardToBitstring(board):
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
    print(len(boardBitstring))

def pieceToBitstring(piece):
    if piece == None:
        return '0000'

    color = '1' if piece.color else '0'
    type = format(piece.piece_type, '03b')

    return color + type



board = first_game.board()
for move in first_game.mainline_moves():
    board.push(move)
    boardToBitstring(board)
