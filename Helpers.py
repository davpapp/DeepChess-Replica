import chess
import chess.pgn

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
