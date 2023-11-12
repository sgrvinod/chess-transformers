import time
import signal
from functools import partial

from chess_transformers.play.exceptions import (
    OutOfTime,
    ClockNotStarted,
    ClockAlreadyStarted,
)


class ChessClock:
    """
    A clock implementing Fischer time control.
    """

    def __init__(
        self,
        base_time,
        increment,
        white_player="White",
        black_player="Black",
        verbose=True,
    ):
        """
        Init.

        Args:

            base_time (int): The base or main time, in seconds.

            increment (int): The time increment per move, in seconds.

            white_player_name (str, optional): The name of the player
            playing white. Defaults to "White".

            black_player_name (str, optional): The name of the player
            playing black. Defaults to "Black".

            verbose (bool, optional): Print clock status on each action?
            Defaults to True.
        """
        self.base_time = base_time
        self.increment = increment
        self.white_player = white_player
        self.black_player = black_player
        self.verbose = verbose

        self._white_time = base_time
        self._black_time = base_time
        self._white_to_move = True
        self._started = False
        self._last_tap = 0

    def start(self):
        """
        Start game, and therefore the clock for white.
        """
        if self._started:
            raise ClockAlreadyStarted

        self._started = True
        if self.verbose:
            print("\nClock started!\n")

        signal.signal(signal.SIGALRM, partial(self._signal_handler, self.white_player))
        signal.setitimer(signal.ITIMER_REAL, self._white_time)
        self._last_tap = time.time()

    def tap(self):
        """
        Tap clock -- by the player who just made a move.
        """
        if not self._started:
            raise ClockNotStarted

        this_tap = time.time()
        signal.setitimer(signal.ITIMER_REAL, 0)

        if self._white_to_move:
            self._white_time -= this_tap - self._last_tap - self.increment

        else:
            self._black_time -= this_tap - self._last_tap - self.increment

        self._white_to_move = not self._white_to_move
        signal.signal(
            signal.SIGALRM,
            partial(
                self._signal_handler,
                self.white_player if self._white_to_move else self.black_player,
            ),
        )
        signal.setitimer(
            signal.ITIMER_REAL,
            self._white_time if self._white_to_move else self._black_time,
        )
        self._last_tap = time.time()
        if self.verbose:
            _, __ = self.status()

    def status(self, verbose=True):
        """
        Show time remaining for both players.

        Args:

            verbose (bool, optional): Print the time remaining in
            addition to returning it? (The clock's verbosity must also
            be set to True during initialization.) Defaults to True.

        Returns:

            float: The time remaining for the player playing white, in
            seconds

            float: The time remaining for the player playing black, in
            seconds
        """
        _white_time = max(
            self._white_time
            - self._white_to_move * self._started * (time.time() - self._last_tap),
            0,
        )
        _black_time = max(
            self._black_time
            - (not self._white_to_move)
            * self._started
            * (time.time() - self._last_tap),
            0,
        )

        if self.verbose and verbose:
            print(
                "\nTime remaining for {}: {:.2f}s".format(
                    self.white_player, _white_time
                ),
            )
            print(
                "Time remaining for {}: {:.2f}s\n".format(
                    self.black_player, _black_time
                )
            )

        return _white_time, _black_time

    def stop(self):
        if self._started:
            signal.setitimer(signal.ITIMER_REAL, 0)

    def reset(self):
        """
        Reset clock.
        """
        self.stop()
        self._white_time = self.base_time
        self._black_time = self.base_time
        self._white_to_move = True
        self._started = False
        self._last_tap = 0

    def _signal_handler(self, player, signum, frame):
        raise OutOfTime(player)
