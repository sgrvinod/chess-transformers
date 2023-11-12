import os
import json
import chess
import torch
from datetime import date
from IPython.display import clear_output, Markdown, display

from chess_transformers.play.utils import (
    get_model_inputs,
    get_legal_moves,
    topk_sampling,
    print_board,
    print_text,
    in_notebook,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_move(model, board, vocabulary, use_amp, k):
    """
    Have the model make the next move on the board.

    Args:

        model (torch.nn.Module): The model.

        board (chess.Board): The chessboard in its current state.

        vocabulary (dict): The vocabulary.

        use_amp (bool): Use automatic mixed precision?

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

    Returns:

        chess.Board: The chessboard after the model makes its move.
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        with torch.autocast(
            device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp
        ):
            model_inputs = get_model_inputs(board, vocabulary)
            predicted_moves = model(
                model_inputs
            )  # (1, max_move_sequence_length, move_vocab_size)
        predicted_moves = predicted_moves[:, 0, :]  # (1, move_vocab_size)

        # Get list of legal moves for the current position
        legal_moves = get_legal_moves(board, vocabulary)

        # Perform top-k sampling to obtain a legal predicted move
        legal_move_index = topk_sampling(
            logits=predicted_moves[
                :, [vocabulary["move_sequence"][m] for m in legal_moves]
            ],
            k=k,
        ).item()
        model_move = legal_moves[legal_move_index]

        # Move
        board.push_uci(model_move.lower())
        clear_output(wait=True)
        display(board) if in_notebook() else print_board(board)

        return board


def engine_move(
    board,
    engine,
    time_limit=None,
    depth_limit=None,
    white_clock=None,
    black_clock=None,
    white_inc=None,
    black_inc=None,
):
    """
    Have an engine make the next move at the given board position.

    Args:

        board (chess.Board): The chessboard in its current state.

        engine (chess.engine.SimpleEngine): The chess engine.

        time_limit (float, optional): Maximum thinking time per move, in
        seconds. Defaults to None.

        depth_limit (int, optional): Maximum depth allowed per move.
        Defaults to None.

        white_clock (float, optional): Time remaining on the clock of
        the player playing white, in seconds. Defaults to None.

        black_clock (float, optional): Time remaining on the clock of
        the player playing black, in seconds. Defaults to None.

        white_inc (float, optional): The increment per move on the clock
        of the player playing white, in seconds. Defaults to None.

        black_inc (float, optional): The increment per move on the clock
        of the player playing black, in seconds. Defaults to None.

    Returns:

        chess.Board: The chessboard after the engine makes its move.
    """
    result = engine.play(
        board,
        chess.engine.Limit(
            time=time_limit,
            depth=depth_limit,
            white_clock=white_clock,
            black_clock=black_clock,
            white_inc=white_inc,
            black_inc=black_inc,
        ),
    )
    stockfish_move = result.move
    board.push(stockfish_move)
    clear_output(wait=True)
    display(board) if in_notebook() else print_board(board)

    return board


def human_move(board):
    """
    Have the human make the next move at the given board position.

    Args:

        board (chess.Board): The chessboard in its current state.

    Returns:

        chess.Board: The chessboard after the human makes their move.

        str, NoneType: The outcome of the move, if the game ends with
        this move.
    """
    legal_moves = [m.uci() for m in board.legal_moves]

    while True:
        human_move = input(
            "What move would you like to play? (UCI notation; 'exit', 'resign', 'draw' are options.)"
        )

        if human_move in legal_moves:
            board.push_uci(human_move)
            clear_output(wait=True)
            msg = " # You played ***%s***." % human_move
            display(Markdown(msg)) if in_notebook() else print_text(msg)
            display(board) if in_notebook() else print_board(board)
            if board.is_checkmate():
                clear_output(wait=True)
                msg = "# You played ***%s***. You win! :(" % human_move
                display(Markdown(msg)) if in_notebook() else print_text(msg)
                display(board) if in_notebook() else print_board(board)
                return board, "0-1" if board.turn else "1-0"
            return board, None

        # If the human wishes to stop playing
        if human_move.lower() == "exit":
            clear_output(wait=True)
            display(Markdown("# You stopped playing."))
            display(board)
            return board, "0-1" if board.turn else "1-0"

        # If the human wishes to resign
        if human_move.lower() == "resign":
            clear_output(wait=True)
            display(Markdown("# You resigned."))
            display(board)
            return board, "0-1" if board.turn else "1-0"

        # If the human wishes to claim a draw
        if human_move.lower() == "draw":
            if board.can_claim_draw():
                clear_output(wait=True)
                display(Markdown("# You claimed a draw."))
                display(board)
                return board, "1/2-1/2"
            else:
                clear_output(wait=True)
                display(Markdown("# You can't claim a draw right now."))
                display(board)

        # If it isn't a legal move
        else:
            clear_output(wait=True)
            display(Markdown("# ***%s*** isn't a valid move." % human_move))
            display(board)
