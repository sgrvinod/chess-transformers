import chess
from IPython.utils import io
from IPython.display import display, Markdown, clear_output

from chess_transformers.play.exceptions import OutOfTime
from chess_transformers.play.moves import model_move, engine_move, human_move
from chess_transformers.play.utils import get_pgn, in_notebook, print_text, print_board


def model_v_engine(
    model,
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
    Have the model play against a chess engine.

    Args:

        model (torch.nn.Module): The model.

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        use_amp (bool): Use automatic mixed precision?

        model_color (str): The color played by the model, one of "w" or
        "b".

        engine (chess.engine.SimpleEngine): The chess engine.

        time_limit (float, optional): Maximum thinking time per move, in
        seconds. Defaults to None.

        depth_limit (int, optional): Maximum depth allowed per move.
        Defaults to None.

        uci_options (dict, optional): UCI options to pass to the chess
        engine. Defaults to an empty dictionary, or no options.

        rounds (int, optional): The number of rounds/games to play.
        Defaults to 1, or a single game.

        clock (chess_transformers.play.clocks.ChessClock, optional): A
        clock set to a time control. Defaults to None, for no time
        control.

        white_player_name (str, optional): The name of the player
        playing white. Defaults to "White".

        black_player_name (str, optional): The name of the player
        playing black. Defaults to "Black".

        event (str, optional): The name of the event this/these game(s)
        are being played at. Defaults to None.

    Returns:

        int: The number of rounds won by the model.

        int: The number of rounds lost by the model.

        int: The number of rounds drawn.

        str: The rounds played in PGN format.
    """
    # Set UCI options
    engine.configure(uci_options)

    # Model color
    model_color = model_color.lower()
    assert model_color in ["w", "b"], "Color played by model must be 'w' or 'b'!"

    # Player names
    model_name = white_player_name if model_color == "w" else black_player_name
    engine_name = white_player_name if model_color == "b" else black_player_name

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
                    use_amp=use_amp,
                    k=k,
                    model_name=model_name,
                    opponent_name=engine_name,
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
                board = engine_move(
                    board=board,
                    engine=engine,
                    time_limit=time_limit,
                    depth_limit=depth_limit,
                    white_clock=time_remaining[0],
                    black_clock=time_remaining[1],
                    white_inc=time_increment,
                    black_inc=time_increment,
                    engine_name=engine_name,
                    opponent_name=model_name,
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
                        use_amp=use_amp,
                        k=k,
                        model_name=model_name,
                        opponent_name=engine_name,
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
    score = "# Final Model Score +%d | -%d | =%d" % (wins, losses, draws)
    display(Markdown(score)) if in_notebook() else print_text(score)
    print("\n")

    return wins, losses, draws, pgns


def model_v_model(
    model_w,
    k_w,
    use_amp_w,
    model_b,
    k_b,
    use_amp_b,
    rounds=1,
    white_player_name="White",
    black_player_name="Black",
    event=None,
):
    """
    Have a model play against another model.

    Args:

        model_w (torch.nn.Module): The model playing white.

        k_w (int): The "k" in "top-k" sampling, for sampling the
        predicted moves of the model playing white.

        use_amp_w (bool): Use automatic mixed precision for the model
        playing white?

        model_b (torch.nn.Module): The model playing black.

        k_b (int): The "k" in "top-k" sampling, for sampling the
        predicted moves of the model playing black.

        use_amp_b (bool): Use automatic mixed precision for the model
        playing black?

        rounds (int, optional): The number of rounds/games to play.
        Defaults to 1, or a single game.

        white_player_name (str, optional): The name of the model playing
        white. Defaults to "White".

        black_player_name (str, optional): The name of the model playing
        black. Defaults to "Black".

        event (str, optional): The name of the event this/these game(s)
        are being played at. Defaults to None.


    Returns:

        int: The number of rounds won by the model playing white.

        int: The number of rounds won by the model playing black.

        int: The number of rounds drawn.

        str: The rounds played in PGN format.
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
                use_amp=use_amp_w,
                k=k_w,
            )
            if not board.is_game_over():
                board = model_move(
                    model=model_b,
                    board=board,
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


def human_v_model(
    human_color,
    model,
    k,
    use_amp,
    rounds=1,
    clock=None,
    white_player_name="White",
    black_player_name="Black",
    event=None,
):
    """
    Play against a model.

    Args:

        human_color (str): The color played by the human, one of "w" or
        "b".

        model (torch.nn.Module): The model.

        k (int): The "k" in "top-k" sampling, for sampling the model's
        predicted moves.

        use_amp (bool): Use automatic mixed precision?

        rounds (int, optional): The number of rounds/games to play.
        Defaults to 1, or a single game.

        clock (chess_transformers.play.clocks.ChessClock, optional): A
        clock set to a time control. Defaults to None, for no time
        control.

        white_player_name (str, optional): The name of the player
        playing white. Defaults to "White".

        black_player_name (str, optional): The name of the player
        playing black. Defaults to "Black".

        event (str, optional): The name of the event this/these game(s)
        are being played at. Defaults to None.

    Returns:

        int: The number of rounds won by the human.

        int: The number of rounds lost by the human.

        int: The number of rounds drawn.

        str: The rounds played in PGN format.
    """
    # Model color
    human_color = human_color.lower()
    assert human_color in ["w", "b"], "Color played by human must be 'w' or 'b'!"

    # Play
    pgns = list()
    wins, draws, losses = 0, 0, 0
    for i in range(rounds):
        time_forfeit = False
        declared_result = None
        board = chess.Board()
        if clock is not None:
            clock.reset()
            clock.start()
        if human_color.lower() == "w":
            try:
                board, declared_result = human_move(
                    board=board,
                    msg="# This is round {} of {}. Your turn...".format(i + 1, rounds),
                )
                if declared_result:
                    if clock is not None:
                        clock.stop()
                    wins += 0
                    losses += 1  # at this point, declared result can only be "0-1" due to resignation or abandonment
                    draws += 0
                    pgns.append(
                        get_pgn(
                            board,
                            white_player_name=white_player_name,
                            black_player_name=black_player_name,
                            event=event,
                            round=str(i + 1),
                            time_control="{}+{}".format(
                                clock.base_time, clock.increment
                            )
                            if clock is not None
                            else "-",
                            result="0-1",
                            termination="normal",
                        )
                    )
                    continue
                if clock is not None:
                    _ = clock.status()
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
                time_forfeit = True
                continue

        while not board.is_game_over():
            try:
                board = model_move(
                    model=model,
                    board=board,
                    use_amp=use_amp,
                    k=k,
                    show_board=False,
                )
                if board.is_checkmate():
                    clear_output(wait=True)
                    msg = "# I played ***%s***. I win! :)" % str(board.move_stack[-1])
                    display(Markdown(msg)) if in_notebook() else print_text(msg)
                    display(board) if in_notebook() else print_board(board)
                elif len(board.move_stack) == 1:
                    msg = "# This is round {} of {}. I played ***{}***. Your turn...".format(
                        i + 1, rounds, str(board.move_stack[-1])
                    )
                else:
                    msg = "# You played ***%s***. I played ***%s***." % (
                        str(board.move_stack[-2]),
                        str(board.move_stack[-1]),
                    )
                if clock is not None:
                    _ = clock.status()
                    clock.tap()
            except OutOfTime as exc:
                display(exc)
                time_forfeit = True
                break
            if not board.is_game_over():
                try:
                    board, declared_result = human_move(board=board, msg=msg)
                    if (
                        declared_result
                    ):  # can be either resignation, abandonment, or a claim of draw
                        if clock is not None:
                            clock.stop()
                        break
                    if clock is not None:
                        _ = clock.status()
                        clock.tap()
                except OutOfTime as exc:
                    display(exc)
                    time_forfeit = True
                    break

        if clock is not None:
            clock.stop()

        if (
            board.result() == "*"
        ):  # can be due to time forfeit or declared result (resignation, abandonment, or claim of draw)
            turn = "w" if board.turn else "b"
            if declared_result == "1/2-1/2":  # claim of draw by human
                draws += 1
            elif human_color != turn:  # time forfeit by model
                wins += 1
                declared_result = "1-0" if human_color == "w" else "0-1"
            else:  # time forfeit or resignation or abandonment by human
                losses += 1
                declared_result = "0-1" if human_color == "w" else "1-0"
        else:
            wins += (
                int(board.result() == "1-0")
                if human_color.lower() == "w"
                else int(board.result() == "0-1")
            )
            losses += (
                int(board.result() == "0-1")
                if human_color.lower() == "w"
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
                result=board.result() if not declared_result else declared_result,
                termination="normal" if not time_forfeit else "time forfeit",
            )
        )

    # Create string of PGNs
    pgns = "\n\n".join(pgns)

    # Print final score
    score = "# Your final Score +%d | -%d | =%d" % (wins, losses, draws)
    display(Markdown(score)) if in_notebook() else print_text(score)
    print("\n")

    return wins, losses, draws, pgns


def warm_up(model):
    """
    Warm up a model by having it play a game against itself.

    This is done to trigger the much-longer-than-usual first run of the
    model, where compilation takes place.

    Args:

        model (torch.nn.Module): The model
    """
    with io.capture_output():
        _ = model_v_model(
            model_w=model,
            k_w=1,
            use_amp_w=True,
            model_b=model,
            k_b=1,
            use_amp_b=True,
        )
    print("\nModel warmed up!")
