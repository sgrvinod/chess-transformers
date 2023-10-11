import time


class ChessClock:
    """
    A clock implementing Fischer time control.

    Note that upon running out of time, a player is alerted only when
    attempting to make a move. Threading is not implemented.
    """

    def __init__(self, base_time, increment, verbose=True):
        """
        Init.

        Args:

            base_time (int): The base or main time, in seconds.

            increment (int): The time increment per move, in seconds.

            verbose (bool, optional): Print clock status on each move?
            Defaults to True.
        """
        self.base_time = base_time
        self.increment = increment
        self._white_time = base_time
        self._black_time = base_time
        self._white_to_move = True
        self.verbose = verbose
        self._started = False
        self._last_tap_white = self._last_tap_black = 0

    def start(self):
        """
        Start game, and therefore the clock for white.
        """
        self._last_tap_white = self._last_tap_black = time.time()
        self._started = True
        if self.verbose:
            print("Clock started!")

    def status(self):
        """
        Print time remaning for both players.
        """
        if self._white_to_move:
            print(
                "Time remaining for White: {:.2f}s".format(
                    self._white_time
                    - self._started * (time.time() - self._last_tap_black)
                )
            )
            print("Time remaining for Black: {:.2f}s".format(self._black_time))
        else:
            print("Time remaining for White: {:.2f}s".format(self._white_time))
            print(
                "Time remaining for Black: {:.2f}s".format(
                    self._black_time
                    - self._started * (time.time() - self._last_tap_white)
                )
            )

    def tap(self):
        """
        Tap clock -- by the player who just made a move.

        Raises:

            TimeoutError: If upon moving it is discovered that the
            player is out of time.
        """
        this_tap = time.time()
        if self._white_to_move:
            self._white_time -= this_tap - self._last_tap_black - self.increment
            self._last_tap_white = this_tap
            if self._white_time < 0:
                raise TimeoutError("White ran out of time!")
            if self.verbose:
                print("Time remaining for White: {:.2f}s".format(self._white_time))
                print("Time remaining for Black: {:.2f}s".format(self._black_time))

        else:
            self._black_time -= this_tap - self._last_tap_white - self.increment
            self._last_tap_black = this_tap
            if self._black_time < 0:
                raise TimeoutError("Black ran out of time!")
            if self.verbose:
                print("Time remaining for White: {:.2f}s".format(self._white_time))
                print("Time remaining for Black: {:.2f}s".format(self._black_time))
        self._white_to_move = not self._white_to_move
