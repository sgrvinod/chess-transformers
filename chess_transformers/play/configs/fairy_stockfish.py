import os

FAIRY_STOCKFISH_PATH = os.environ.get(
    "CT_FAIRY_STOCKFISH_PATH"
)  # path to Fairy Stockfish engine
LICHESS_LEVELS = {
    1: {"SKILL": -9, "DEPTH": 5, "TIME_CONSTRAINT": 0.050},
    2: {"SKILL": -5, "DEPTH": 5, "TIME_CONSTRAINT": 0.100},
    3: {"SKILL": -1, "DEPTH": 5, "TIME_CONSTRAINT": 0.150},
    4: {"SKILL": 3, "DEPTH": 5, "TIME_CONSTRAINT": 0.200},
    5: {"SKILL": 7, "DEPTH": 5, "TIME_CONSTRAINT": 0.300},
    6: {"SKILL": 11, "DEPTH": 8, "TIME_CONSTRAINT": 0.400},
    7: {"SKILL": 16, "DEPTH": 13, "TIME_CONSTRAINT": 0.500},
    8: {"SKILL": 20, "DEPTH": 22, "TIME_CONSTRAINT": 1.000},
}  # from https://github.com/lichess-org/fishnet/blob/dc4be23256e3e5591578f0901f98f5835a138d73/src/api.rs#L224
