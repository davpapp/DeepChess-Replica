from play import find_best_move
import chess
import chess.pgn
# app.py
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

@app.route('/hello', methods=['GET', 'POST'])
def hello():

    # POST request
    if request.method == 'POST':
        print('Incoming..')
        board = chess.Board(fen=request.get_json()['fen'])
        print(board)
        result = find_best_move(board, True)
        print(result)
        #print(request.get_json())  # parse as JSON
        return str(result), 200

    # GET request
    else:
        message = {'greeting':'Hello from Flask!'}
        return jsonify(message)  # serialize and use JSON headers

@app.route('/test')
def test_page():
    # look inside `templates` and serve `index.html`
    return render_template('index.html')
