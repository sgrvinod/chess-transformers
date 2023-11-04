import os
import sys
import json
import chess
import markdown
import textwrap
import chess.pgn
import chess.engine
import torch.utils.data
from datetime import date
from tabulate import tabulate
from bs4 import BeautifulSoup
from IPython.display import display
from contextlib import contextmanager
from colorama import Fore, Back, Style

from chess_transformers.data.utils import encode, parse_fen


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

UNICODE_CHESS_PIECES = {
    "R": Fore.WHITE + "♜",
    "N": Fore.WHITE + "♞",
    "B": Fore.WHITE + "♝",
    "Q": Fore.WHITE + "♛",
    "K": Fore.WHITE + "♚",
    "P": Fore.WHITE + "♟︎",
    "r": Fore.BLACK + "♜",
    "n": Fore.BLACK + "♞",
    "b": Fore.BLACK + "♝",
    "q": Fore.BLACK + "♛",
    "k": Fore.BLACK + "♚",
    "p": Fore.BLACK + "♟︎",
    ".": " ",
}


def load_assets(CONFIG):
    # Model
    _model = CONFIG.MODEL(CONFIG).to(DEVICE)

    # Checkpoint
    checkpoint = torch.load(
        os.path.join(CONFIG.CHECKPOINT_FOLDER, CONFIG.FINAL_CHECKPOINT)
    )
    _model.load_state_dict(checkpoint["model_state_dict"])

    # Compile model
    model = torch.compile(
        _model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )
    model.eval()  # eval mode disables dropout

    # Vocabulary
    vocabulary = json.load(open(os.path.join(CONFIG.DATA_FOLDER, CONFIG.VOCAB_FILE)))
    for vocabulary_name in vocabulary:
        if set(vocabulary[vocabulary_name].keys()).issubset({"true", "false"}):
            for bool_string in set(vocabulary[vocabulary_name].keys()):
                index = vocabulary[vocabulary_name].pop(bool_string)
                vocabulary[vocabulary_name][bool_string == "true"] = index

    return model, vocabulary


def load_engine(path):
    engn = chess.engine.SimpleEngine.popen_uci(path)

    # Show options
    print("Engine loaded with UCI options:\n")
    table = list()
    for item, option in dict(engn.options).items():
        table.append(
            [
                item,
                option.default,
                option.min,
                option.max,
                "\n".join(textwrap.wrap(", ".join(option.var)))
                if len(option.var) > 0
                else None,
            ]
        )
    display(
        tabulate(
            table,
            headers=["Name", "Default", "Minimum", "Maximum", "Variants"],
            tablefmt="html" if "ipykernel" in sys.modules else "fancy_grid",
            colalign=["center", "center", "center", "center", "center"],
        )
    )

    return engn


def get_model_inputs(board, vocabulary):
    model_inputs = dict()

    t, b, wk, wq, bk, bq = parse_fen(board.fen())
    model_inputs["turns"] = (
        torch.IntTensor([encode(t, vocabulary=vocabulary["turn"])])
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["board_positions"] = (
        torch.IntTensor(encode(b, vocabulary=vocabulary["board_position"]))
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["white_kingside_castling_rights"] = (
        torch.IntTensor(
            [encode(wk, vocabulary=vocabulary["white_kingside_castling_rights"])]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["white_queenside_castling_rights"] = (
        torch.IntTensor(
            [encode(wq, vocabulary=vocabulary["white_queenside_castling_rights"])]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["black_kingside_castling_rights"] = (
        torch.IntTensor(
            [encode(bk, vocabulary=vocabulary["black_kingside_castling_rights"])]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["black_queenside_castling_rights"] = (
        torch.IntTensor(
            [encode(bq, vocabulary=vocabulary["black_queenside_castling_rights"])]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["moves"] = (
        torch.LongTensor(
            [
                vocabulary["move_sequence"]["<move>"],
                vocabulary["move_sequence"]["<pad>"],
            ]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["lengths"] = torch.LongTensor([1]).unsqueeze(0).to(DEVICE)

    return model_inputs


def get_legal_moves(board, vocabulary):
    legal_moves = [
        str(m) for m in board.legal_moves if str(m) in vocabulary["move_sequence"]
    ]

    return legal_moves


def topk_sampling(logits, k=5):
    """
    Randomly sample from the multinomial distribution formed by the
    "top-k" logits only.

    Args:

        logits (torch.FloatTensor): Predicted next-move probabilities,
        of size (N, move_vocab_size).

        k (int, optional): Value of "k". Defaults to 5.

    Returns:

        torch.LongTensor: Samples (indices), of size (N).
    """
    k = min(k, logits.shape[1])

    # Find the kth-highest logit value per row
    max_logit_values = logits.topk(k=k, dim=1)[0][:, -1:]  # (N, 1)

    # All other logit values must be ignored; they should evaluate to 0
    # under a softmax op.
    logits[logits < max_logit_values] = -float("inf")  #  (N, move_vocab_size)

    # Apply softmax
    probabilities = torch.softmax(logits, dim=1)  #  (N, move_vocab_size)

    # Sample from this multinomial probability distribution
    samples = torch.multinomial(probabilities, num_samples=1).squeeze(1)  #  (N)

    return samples


def get_pgn(
    board,
    white_player_name=None,
    black_player_name=None,
    event=None,
    round=None,
    time_control="?",
    result=None,
    termination=None,
):
    game = chess.pgn.Game.from_board(board)

    if white_player_name is not None:
        game.headers["White"] = white_player_name
    if black_player_name is not None:
        game.headers["Black"] = black_player_name

    game.headers["Date"] = date.today().strftime("%Y/%m/%d")

    if event is not None:
        game.headers["Event"] = event

    game.headers["Site"] = "github.com/sgrvinod/chess-transformers"

    if round is not None:
        game.headers["Round"] = str(round)

    game.headers["TimeControl"] = "?" if time_control in [None, "?"] else time_control

    game.headers["Result"] = result if result else board.result()

    if termination is not None:
        game.headers["Termination"] = termination
    elif game.headers["Result"] in ["1-0", "0-1", "1/2-1/2"]:
        game.headers["Termination"] = "normal"
    else:
        game.headers["Termination"] = "unterminated"

    return str(game)


def print_board(board):
    # Get coordinates (flattened index) for the "from" and "to" squares of the last move
    last_move = board.peek()
    from_rank_idx = chess.square_rank(last_move.from_square)
    from_file_idx = chess.square_file(last_move.from_square)
    from_square_idx = (8 - (from_rank_idx + 1)) * 8 + from_file_idx
    to_rank_idx = chess.square_rank(last_move.to_square)
    to_file_idx = chess.square_file(last_move.to_square)
    to_square_idx = (8 - (to_rank_idx + 1)) * 8 + to_file_idx

    # Check for check or checkmate
    is_check = board.is_check()
    is_checkmate = board.is_checkmate()
    turn = "w" if board.turn else "b"

    # "Render" chessboard
    board_str = str(board).replace(" ", "")
    new_board = "\n    a  b  c  d  e  f  g  h \n 8 "
    white_bg = True
    file = 8
    sq_counter = 0
    for p in board_str:
        if p != "\n":
            if sq_counter not in {from_square_idx, to_square_idx}:
                bg = Back.CYAN if white_bg else Back.BLUE
            else:
                bg = Back.GREEN
            if turn == "w" and p == "K" and is_checkmate:
                bg = Back.RED
            elif turn == "b" and p == "k" and is_checkmate:
                bg = Back.RED
            elif turn == "w" and p == "K" and is_check:
                bg = Back.YELLOW
            elif turn == "b" and p == "k" and is_check:
                bg = Back.YELLOW
            new_board += bg + " " + Style.BRIGHT + UNICODE_CHESS_PIECES[p] + " "
            white_bg = not white_bg
            sq_counter += 1
        else:
            file -= 1
            new_board += Style.RESET_ALL + " " + "\n" + " " + str(file) + " "
            white_bg = not white_bg
    print(new_board + Style.RESET_ALL + "\n")
    if is_checkmate:
        print("CHECKMATE!")
    elif is_check:
        print("Check!")


def in_notebook():
    try:
        from IPython import get_ipython

        if "IPKernelApp" not in get_ipython().config:
            return False
    except ImportError:
        return False
    except AttributeError:
        return False
    return True


def print_text(md):
    html = markdown.markdown(md)
    text = "\n".join(BeautifulSoup(html, features="lxml").findAll(text=True))

    print(text)


def write_pgns(pgns, pgn_file):
    with open(pgn_file, "w") as f:
        f.write(pgns)


@contextmanager
def suppress_stdouterr():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
