import os
import json
import chess
import random
import chess.pgn
import numpy as np
import tables as tb
from tqdm import tqdm
from collections import Counter


def view_game(game):
    board = game.board()
    for move_number, move in enumerate(game.mainline_moves()):
        print("\n")
        print("Move #%d" % move_number)
        print("LAN:", board.lan(move))
        print("SAN:", board.san(move))
        print("UCI:", board.uci(move))
        board.push(move)
        print(board)
