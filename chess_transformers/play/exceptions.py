class OutOfTime(Exception):
    """
    An exception to raise when a player runs out of time on their clock.
    """

    def __init__(self, player, message="Player ran out of time!"):
        self.player = player
        self.message = message

    def __str__(self):
        return self.message.replace("Player", self.player)

    def __call__(self):
        raise self


class ClockNotStarted(Exception):
    """
    An exception to raise when an action requires the clock to have
    already been started, but it has not yet been started.
    """

    def __init__(self, message="Clock has not been started!"):
        self.message = message

    def __str__(self):
        return self.message


class ClockAlreadyStarted(Exception):
    """
    An exception to raise when an action requires the clock to have NOT
    been started, but it already has been started.
    """

    def __init__(self, message="Clock has already been started!"):
        self.message = message

    def __str__(self):
        return self.message
