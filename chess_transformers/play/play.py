import chess
from IPython.utils import io
from IPython.display import display, Markdown

from chess_transformers.play.utils import (
    get_pgn,
    in_notebook,
    print_text,
)
from chess_transformers.play.exceptions import OutOfTime
from chess_transformers.play.moves import model_move, stockfish_move


def model_v_engine(
    model,
    vocabulary,
    k,
    use_amp,
    model_color,
    engine,
    time_limit=None,
    depth_limit=None,
    uci_options=dict(),
    rounds=1,
    clock=None,
    white_player_name="White",
    black_player_name="Black",
    event=None,
):
    """
    Have the model play a game against the Stockfish engine.

    Args:

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        engine (chess.engine.SimpleEngine): The Stockfish engine.

        time_limit (int): Maximum thinking time per move, in seconds.

        depth_limit (int): Maximum depth allowed per move.

    Returns:

        chess.Board: The played-out chessboard at the end of the game.
    """
    # Set UCI options
    engine.configure(uci_options)

    # Model color
    model_color = model_color.lower()
    assert model_color in ["w", "b"], "Model color must be 'w' or 'b'!"

    # Play
    pgns = list()
    wins, draws, losses = 0, 0, 0
    for i in range(rounds):
        board = chess.Board()
        if clock is not None:
            clock.reset()
            clock.start()
        if model_color.lower() == "w":
            try:
                board = model_move(
                    model=model,
                    board=board,
                    vocabulary=vocabulary,
                    use_amp=use_amp,
                    k=k,
                )
                if clock is not None:
                    clock.tap()
            except OutOfTime as exc:
                display(exc)
                wins += 0
                losses += 1
                draws += 0
                pgns.append(
                    get_pgn(
                        board,
                        white_player_name=white_player_name,
                        black_player_name=black_player_name,
                        event=event,
                        round=str(i + 1),
                        time_control="{}+{}".format(clock.base_time, clock.increment)
                        if clock is not None
                        else "-",
                        result="0-1",
                        termination="time forfeit",
                    )
                )
                continue

        while not board.is_game_over():
            try:
                time_remaining = clock.status(verbose=False) if clock else (None, None)
                time_increment = clock.increment if clock else None
                board = stockfish_move(
                    board=board,
                    engine=engine,
                    time_limit=time_limit,
                    depth_limit=depth_limit,
                    white_clock=time_remaining[0],
                    black_clock=time_remaining[1],
                    white_inc=time_increment,
                    black_inc=time_increment,
                )
                if clock is not None:
                    clock.tap()
            except OutOfTime as exc:
                display(exc)
                break
            if not board.is_game_over():
                try:
                    board = model_move(
                        model=model,
                        board=board,
                        vocabulary=vocabulary,
                        use_amp=use_amp,
                        k=k,
                    )
                    if clock is not None:
                        clock.tap()
                except OutOfTime as exc:
                    display(exc)
                    break

        if clock is not None:
            clock.stop()

        time_forfeit_result = None
        if board.result() == "*":
            turn = "w" if board.turn else "b"
            if model_color != turn:
                wins += 1
                time_forfeit_result = "1-0" if model_color == "w" else "0-1"
            else:
                losses += 1
                time_forfeit_result = "0-1" if model_color == "w" else "1-0"
        else:
            wins += (
                int(board.result() == "1-0")
                if model_color.lower() == "w"
                else int(board.result() == "0-1")
            )
            losses += (
                int(board.result() == "0-1")
                if model_color.lower() == "w"
                else int(board.result() == "1-0")
            )
            draws += int(board.result() == "1/2-1/2")

        # PGN
        pgns.append(
            get_pgn(
                board,
                white_player_name=white_player_name,
                black_player_name=black_player_name,
                event=event,
                round=str(i + 1),
                time_control="{}+{}".format(clock.base_time, clock.increment)
                if clock is not None
                else "-",
                result=board.result()
                if not time_forfeit_result
                else time_forfeit_result,
                termination="normal" if not time_forfeit_result else "time forfeit",
            )
        )

    # Create string of PGNs
    pgns = "\n\n".join(pgns)

    # Print final score
    print("\n")
    score = "# Final Model Score +%d | -%d | =%d" % (wins, losses, draws)
    display(Markdown(score)) if in_notebook() else print_text(score)
    print("\n")

    return wins, losses, draws, pgns


def model_v_model(
    model_w,
    vocabulary_w,
    k_w,
    use_amp_w,
    model_b,
    vocabulary_b,
    k_b,
    use_amp_b,
    rounds=1,
    white_player_name="White",
    black_player_name="Black",
    event=None,
):
    """
    Have the model play a game against the Stockfish engine.

    Args:

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        engine (chess.engine.SimpleEngine): The Stockfish engine.

        time_limit (int): Maximum thinking time per move, in seconds.

        depth_limit (int): Maximum depth allowed per move.

    Returns:

        chess.Board: The played-out chessboard at the end of the game.
    """
    # Play
    pgns = list()
    w_wins, b_wins, draws = 0, 0, 0
    for i in range(rounds):
        board = chess.Board()
        while not board.is_game_over():
            board = model_move(
                model=model_w,
                board=board,
                vocabulary=vocabulary_w,
                use_amp=use_amp_w,
                k=k_w,
            )
            board = model_move(
                model=model_b,
                board=board,
                vocabulary=vocabulary_b,
                use_amp=use_amp_b,
                k=k_b,
            )
        w_wins += int(board.result() == "1-0")
        b_wins += int(board.result() == "0-1")
        draws += int(board.result() == "1/2-1/2")

        # PGN
        pgns.append(
            get_pgn(
                board,
                white_player_name=white_player_name,
                black_player_name=black_player_name,
                event=event,
                round=str(i + 1),
                time_control="-",
                result=board.result(),
            )
        )

    # Create string of PGNs
    pgns = "\n\n".join(pgns)

    # Print final score
    print("\n")
    score = "# Final Score: %s %d | %s %d | Draws %d" % (
        white_player_name,
        w_wins,
        black_player_name,
        b_wins,
        draws,
    )
    display(Markdown(score)) if in_notebook() else print_text(score)
    print("\n")

    return w_wins, b_wins, draws, pgns


def warm_up(model, vocabulary):
    with io.capture_output():
        _ = model_v_model(
            model_w=model,
            vocabulary_w=vocabulary,
            k_w=1,
            use_amp_w=True,
            model_b=model,
            vocabulary_b=vocabulary,
            k_b=1,
            use_amp_b=True,
        )
    print("\nModel warmed up!")
