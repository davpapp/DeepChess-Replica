import chess
import chess.pgn
import pickle

from Helpers import boardToBitstring

pgn = open('games/2015-05.bare.[6004].pgn')
PARSED_GAME_FILE = 'parsed_games/2015-05.bare.[6004].parsed.pickle'
PARSED_FLATTENED_GAME_FILE = 'parsed_games/2015-05.bare.[6004].parsed_flattened.pickle'

def parse_game(game):
    board = game.board()
    board_data = []
    boards = []
    #print(game.mainline_moves())
    move_number = 1
    alg_move = ""
    for move in game.mainline_moves():
        if move_number > 1:
            alg_move = board.san(move)
        if not move:
            break
        board.push(move)
        move_number += 1
        if move_number > 5 and move_number % 5 == 0 and 'x' not in alg_move:
            board_data.append(boardToBitstring(board))
            boards.append(board.fen())
    #print(boards)
    return [board_data, boards, game.headers['Result']]



data = []
idx = 0
while True:
    if idx > 100000:
        break
    game = chess.pgn.read_game(pgn)
    if not game:
        break
    if game.headers['Result'] == '1/2-1/2':
        #print("Skipping tie...")
        continue
    data.append(parse_game(game))
    idx += 1

# There's probably a cleaner way of doing this.
flattened_data = []
for game in data:
    boards_data, boards, outcome = game[0], game[1], game[2]

    for idx in range(0, len(boards_data)):
        flattened_data.append([boards_data[idx], boards[idx], outcome])

print(flattened_data[:5])

with open(PARSED_GAME_FILE, 'wb') as handle:
    pickle.dump(data, handle)

with open(PARSED_FLATTENED_GAME_FILE, 'wb') as handle:
    pickle.dump(flattened_data, handle)

#print(data[2000])




#board = first_game.board()
#for move in first_game.mainline_moves():
#    board.push(move)
#    boardToBitstring(board)
