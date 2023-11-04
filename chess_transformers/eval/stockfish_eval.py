import os
from importlib import import_module

from chess_transformers.play.play import model_v_engine, warm_up
from chess_transformers.play.utils import load_assets, load_engine, write_pgns

uci_elo = 3111.2
e = (uci_elo - 1320) / (3190 - 1320)
s = (((37.2473 * e - 40.8525) * e + 22.2943) * e - 0.311438)
print(s)