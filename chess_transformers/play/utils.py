import os
import sys
import chess
import urllib
import pathlib
import markdown
import textwrap
import chess.pgn
import chess.engine
import torch.utils.data
import torch.nn.functional as F
from tqdm import tqdm
from datetime import date
from tabulate import tabulate
from bs4 import BeautifulSoup
from IPython.display import display
from contextlib import contextmanager
from colorama import Fore, Back, Style

from chess_transformers.data.utils import encode, parse_fen
from chess_transformers.data.levels import TURN, PIECES, UCI_MOVES, BOOL


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


def download_file(url, output_path):
    """
    Download the file at the given URL into a specified output path.

    Adapted from https://github.com/tqdm/tqdm#hooks-and-callbacks

    Args:

        url (str): The URL for the file to download.

        output_path (str): The path of the file to download into.
    """

    class TQDMUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            return self.update(b * bsize - self.n)

    with TQDMUpTo(
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        miniters=1,
        desc=url.split("/")[-1],
    ) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)
        t.total = t.n


def load_model(CONFIG):
    """
    Load model for inference.

    Args:

        CONFIG (dict): The configuration of the model.

    Returns:

        torch.nn.Module: The model.
    """
    # Model
    _model = CONFIG.MODEL(CONFIG).to(DEVICE)

    # Download checkpoint and vocabulary if they haven't already been
    # downloaded
    checkpoint_folder = (
        pathlib.Path(__file__).parent.parent.resolve() / "checkpoints" / CONFIG.NAME
    )
    checkpoint_folder.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_folder / CONFIG.FINAL_CHECKPOINT
    if not checkpoint_path.exists():
        print("\nCannot find model checkpoint on disk; will download.\n")
        download_file(
            "ht"
            + "tp"
            + "s:"
            + "//"
            + "chess"
            + "transformers"
            + "."
            + "blob"
            + "."
            + "core"
            + "."
            + "windows"
            + "."
            + "net"
            + "/checkpoints/{}/{}".format(CONFIG.NAME, CONFIG.FINAL_CHECKPOINT),
            str(checkpoint_path),
        )  # scramble address against simple bots

    # Load checkpoint
    checkpoint = torch.load(str(checkpoint_path), weights_only=True)
    _model.load_state_dict(checkpoint["model_state_dict"])

    # Compile model
    model = torch.compile(
        _model,
        mode=CONFIG.COMPILATION_MODE,
        dynamic=CONFIG.DYNAMIC_COMPILATION,
        disable=CONFIG.DISABLE_COMPILATION,
    )
    model.eval()  # eval mode disables dropout

    print("\nModel loaded!\n")

    return model


def show_engine_options(engine):
    """
    Display available UCI configuration options for a chess engine.

    Args:

        engine (chess.engine.SimpleEngine): The chess engine.
    """
    print("\nUCI options for chess engine:\n")
    table = list()
    for item, option in dict(engine.options).items():
        table.append(
            [
                item,
                option.default,
                option.min,
                option.max,
                (
                    "\n".join(textwrap.wrap(", ".join(option.var)))
                    if len(option.var) > 0
                    else None
                ),
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


def load_engine(path, show_options=True):
    """
    Load a chess engine.

    Args:

        path (str): The path to the engine file (an executable).

        show_options (bool, optional): Print configurable UCI options
        for engine.

    Returns:

        chess.engine.SimpleEngine: The chess engine.
    """
    engine = chess.engine.SimpleEngine.popen_uci(path)

    if show_options:
        show_engine_options(engine)

    return engine


def get_model_inputs(board):
    """
    Get inputs to be fed to a model.

    Args:

        board (chess.Board): The chessboard in its current state.

    Returns:

        dict: The inputs to be fed to the model.
    """
    model_inputs = dict()

    t, b, wk, wq, bk, bq = parse_fen(board.fen())
    model_inputs["turns"] = (
        torch.IntTensor([encode(t, vocabulary=TURN)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["board_positions"] = (
        torch.IntTensor(encode(b, vocabulary=PIECES)).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_kingside_castling_rights"] = (
        torch.IntTensor([encode(wk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["white_queenside_castling_rights"] = (
        torch.IntTensor([encode(wq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_kingside_castling_rights"] = (
        torch.IntTensor([encode(bk, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["black_queenside_castling_rights"] = (
        torch.IntTensor([encode(bq, vocabulary=BOOL)]).unsqueeze(0).to(DEVICE)
    )
    model_inputs["moves"] = (
        torch.LongTensor(
            [
                UCI_MOVES["<move>"],
                UCI_MOVES["<pad>"],
            ]
        )
        .unsqueeze(0)
        .to(DEVICE)
    )
    model_inputs["lengths"] = torch.LongTensor([1]).unsqueeze(0).to(DEVICE)

    return model_inputs


def is_pawn_promotion(board, move):
    """
    Check if a move (in UCI notation) corresponds to a pawn promotion on
    the given board.

    Args:

        board (chess.Board): The chessboard in its current state.

        move (str): The move un UCI notation.

    Returns:

        bool: Is the move a pawn promotion?
    """

    m = chess.Move.from_uci(move)
    if board.piece_type_at(m.from_square) == chess.PAWN and chess.square_rank(
        m.to_square
    ) in [0, 7]:
        return True
    else:
        return False


def topk_sampling(logits, k=1):
    """
    Randomly sample from the multinomial distribution formed from the
    "top-k" logits only.

    Args:

        logits (torch.FloatTensor): Predicted logits, of size (N,
        vocab_size).

        k (int, optional): Value of "k". Defaults to 1.

    Returns:

        torch.LongTensor: Samples (indices), of size (N).
    """
    k = min(k, logits.shape[1])

    with torch.no_grad():
        min_topk_logit_values = logits.topk(k=k, dim=1)[0][:, -1:]  # (N, 1)
        logits[logits < min_topk_logit_values] = -float("inf")  #  (N, vocab_size)
        probabilities = F.softmax(logits, dim=1)  #  (N, vocab_size)
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
    """
    Create the portable game notation (PGN) for a game represented by a
    played-out chessboard.

    Args:

        board (chess.Board): The chessboard in its (presumably)
        played-out state.

        white_player_name (str, optional): The name of the player
        playing white. Defaults to None, in which case this tag is not
        included in the PGN.

        black_player_name (str, optional): The name of the player
        playing black. Defaults to None, in which case this tag is not
        included in the PGN.

        event (str, optional): The name of the event at which the game
        is being played. Defaults to None, in which case this tag is not
        included in the PGN.

        round (str, optional): The round number/name. Defaults to None,
        in which case this tag is not included in the PGN.

        time_control (str, optional): The time control used for this
        game. Defaults to "?", for unknown time control.

        result (str, optional): The result of the game. Defaults to
        None, in which case the result is inferred from the board.

        termination (str, optional): The termination type for the game.
        Defaults to None, in which case the termination is inferred from
        the board as either "normal" or "unterminated".

    Returns:

        str: The PGN for this game.
    """
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
    """
    Display a visual representation of the chessboard in its current
    state in a terminal, as an alternative to python-chess's native
    printing.

    Args:

        board (chess.Board): The chessboard in its current state.
    """
    # Get coordinates (flattened index) for the "from" and "to" squares
    # of the last move
    try:
        last_move = board.peek()
        from_rank_idx = chess.square_rank(last_move.from_square)
        from_file_idx = chess.square_file(last_move.from_square)
        from_square_idx = (8 - (from_rank_idx + 1)) * 8 + from_file_idx
        to_rank_idx = chess.square_rank(last_move.to_square)
        to_file_idx = chess.square_file(last_move.to_square)
        to_square_idx = (8 - (to_rank_idx + 1)) * 8 + to_file_idx
    except IndexError:
        from_square_idx, to_square_idx = None, None

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
    """
    Is this function being called in an IPython notebook, as opposed to
    a terminal?

    Returns:

        bool: Is it?
    """
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
    """
    Print markdown-formatted text as plain text in a terminal.

    Args:

        md (str): The markdown-formatted text.
    """
    html = markdown.markdown(md)
    text = "\n" + "".join(BeautifulSoup(html, features="lxml").findAll(text=True))

    print(text)


def write_pgns(pgns, pgn_file):
    """
    Write PGNs to a file.

    Args:

        pgns (str): The PGNs.

        pgn_file (str): The path to write as a file.
    """
    parent_folder = pathlib.Path(pgn_file).parent.resolve()
    parent_folder.mkdir(parents=True, exist_ok=True)
    with open(pgn_file, "w") as f:
        f.write(pgns)


@contextmanager
def suppress_stdouterr():
    """
    A context manager for suppressing standard output (stdout) and
    standard error (stderr).
    """
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
