import chess
import torch
from IPython.display import clear_output, Markdown, display

from chess_transformers.play.utils import (
    get_model_inputs,
    is_pawn_promotion,
    topk_sampling,
    print_board,
    print_text,
    in_notebook,
)
from chess_transformers.data.levels import SQUARES, UCI_MOVES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def model_move(
    model,
    board,
    use_amp,
    k,
    model_name=None,
    opponent_name=None,
    show_board=True,
):
    """
    Have the model make the next move on the board.

    Args:

        config_name (str): The name of the model configuration (which
        describes the type of model).

        model (torch.nn.Module): The model.

        board (chess.Board): The chessboard in its current state.

        use_amp (bool): Use automatic mixed precision?

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        model_name (str, optional): The name of the model, for
        displaying in status messages. Defaults to None.

        opponent_name (str, optional): The name of the model's opponent,
        for displaying in status messages. Defaults to None.

        show_board (bool, optional): Display the board (along with a
        status message) upon making the move?

    Returns:

        chess.Board: The chessboard after the model makes its move.
    """
    # Get predictions
    model.eval()
    with torch.no_grad():
        # Get list of legal moves for the current position
        legal_moves = [move.uci() for move in board.legal_moves]

        # Get model inputs
        model_inputs = get_model_inputs(board)

        # (Direct) Move prediction models
        if model.code in {"E", "ED"}:
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted_moves = model(model_inputs)
            predicted_moves = predicted_moves[:, 0, :]  # (1, move_vocab_size)

            # Filter out move indices corresponding to illegal moves
            legal_move_indices = [UCI_MOVES[m] for m in legal_moves]

            # Perform top-k sampling to obtain a legal predicted move
            legal_move_index = topk_sampling(
                logits=predicted_moves[:, legal_move_indices],
                k=k,
            ).item()
            model_move = legal_moves[legal_move_index]

        # "From" and "To" square prediction models
        elif model.code in {"EFT"}:
            with torch.autocast(
                device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp
            ):
                predicted_from_squares, predicted_to_squares = model(
                    model_inputs
                )  # (1, 1, 64), (1, 1, 64)
            predicted_from_squares = predicted_from_squares[:, 0, :]  # (1, 64)
            predicted_to_squares = predicted_to_squares[:, 0, :]  # (1, 64)

            # Convert "From" and "To" square predictions to move predictions
            predicted_from_log_probabilities = torch.log_softmax(
                predicted_from_squares, dim=-1
            ).unsqueeze(
                2
            )  # (1, 64, 1)
            predicted_to_log_probabilities = torch.log_softmax(
                predicted_to_squares, dim=-1
            ).unsqueeze(
                1
            )  # (1, 1, 64)
            predicted_moves = (
                predicted_from_log_probabilities + predicted_to_log_probabilities
            ).view(
                1, -1
            )  # (1, 64 * 64)

            # Filter out move indices corresponding to illegal moves
            legal_moves = list(
                set([m[:4] for m in legal_moves])
            )  # for handing pawn promotions manually, remove pawn promotion targets
            legal_move_indices = list()
            for m in legal_moves:
                from_square = m[:2]
                to_square = m[2:4]
                legal_move_indices.append(
                    SQUARES[from_square] * 64 + SQUARES[to_square]
                )

            # Perform top-k sampling to obtain a legal predicted move
            legal_move_index = topk_sampling(
                logits=predicted_moves[:, legal_move_indices],
                k=k,
            ).item()
            model_move = legal_moves[legal_move_index]

            # Handle pawn promotion manually if "model_move" is a pawn promotion move
            if is_pawn_promotion(board, model_move):
                model_move = model_move + "q"  # always promote to a queen

        # Other models
        else:
            raise NotImplementedError

        # Move
        board.push_uci(model_move)
        if show_board:
            clear_output(wait=True)
            if opponent_name and len(board.move_stack) > 1:
                msg = "# {} played ***{}***. {} plays ***{}***.".format(
                    opponent_name,
                    board.move_stack[-2],
                    model_name if model_name else "Model",
                    board.move_stack[-1],
                )
            else:
                msg = "# {} plays ***{}***.".format(
                    model_name if model_name else "Model", board.move_stack[-1]
                )
            display(Markdown(msg)) if in_notebook() else print_text(msg)
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
    engine_name=None,
    opponent_name=None,
    show_board=True,
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

        engine_name (str, optional): The name of the engine, for
        displaying in status messages. Defaults to None.

        opponent_name (str, optional): The name of the engine's
        opponent, for displaying in status messages. Defaults to None.

        show_board (bool, optional): Display the board (along with a
        status message) upon making the move?

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
    engine_move = result.move
    board.push(engine_move)
    if show_board:
        clear_output(wait=True)
        if opponent_name and len(board.move_stack) > 1:
            msg = "# {} played ***{}***. {} plays ***{}***.".format(
                opponent_name,
                board.move_stack[-2],
                engine_name if engine_name else "Engine",
                board.move_stack[-1],
            )
        else:
            msg = "# {} plays {}.".format(
                engine_name if engine_name else "Model", board.move_stack[-1]
            )
        display(Markdown(msg)) if in_notebook() else print_text(msg)
        display(board) if in_notebook() else print_board(board)

    return board


def human_move(
    board,
    msg,
):
    """
    Have the human make the next move at the given board position.

    Args:

        board (chess.Board): The chessboard in its current state.

        msg (chess.Board): A message to display with the chessboard.

    Returns:

        chess.Board: The chessboard after the human makes their move.

        str, NoneType: A declared result due to an end to the game being
        declared by the human, such as when the human abandons, resigns,
        or claims a draw. None if none.
    """
    legal_moves = [m.uci() for m in board.legal_moves]
    while True:
        clear_output(wait=True)
        display(Markdown(msg)) if in_notebook() else print_text(msg)
        display(board) if in_notebook() else print_board(board)
        human_move = input(
            "What move would you like to play? Enter in UCI notation, or 'exit' / 'resign' / 'draw': "
        )
        if human_move in legal_moves:
            board.push_uci(human_move)
            if in_notebook():
                clear_output(wait=True)
                msg = "# You played ***{}***.".format(human_move)
                display(Markdown(msg)) if in_notebook() else print_text(msg)
                display(board) if in_notebook() else print_board(board)
            if board.is_checkmate():
                clear_output(wait=True)
                msg = "# You played ***{}***. You win! :(".format(human_move)
                display(Markdown(msg)) if in_notebook() else print_text(msg)
                display(board) if in_notebook() else print_board(board)
            return board, None

        # If the human wishes to stop playing
        elif human_move.lower() == "exit":
            clear_output(wait=True)
            msg = "# You stopped playing."
            display(Markdown(msg)) if in_notebook() else print_text(msg)
            display(board) if in_notebook() else print_board(board)
            return board, "0-1" if board.turn else "1-0"

        # If the human wishes to resign
        elif human_move.lower() == "resign":
            clear_output(wait=True)
            msg = "# You resigned."
            display(Markdown(msg)) if in_notebook() else print_text(msg)
            display(board) if in_notebook() else print_board(board)
            return board, "0-1" if board.turn else "1-0"

        # If the human wishes to claim a draw
        elif human_move.lower() == "draw":
            clear_output(wait=True)
            if board.can_claim_draw():
                msg = "# You claimed a draw."
                display(Markdown(msg)) if in_notebook() else print_text(msg)
                display(board) if in_notebook() else print_board(board)
                return board, "1/2-1/2"
            else:
                msg = "# You can't claim a draw right now."

        # If it isn't a legal move
        else:
            clear_output(wait=True)
            msg = "# ***%s*** isn't a valid move." % human_move
