

class OutOfTime(Exception):
    def __init__(self, player, message="Player ran out of time!"):
        self.player = player
        self.message = message

    def __str__(self):
        return self.message.replace("Player", self.player)

    def __call__(self):
        raise self

class ClockNotStarted(Exception):
    def __init__(self, message="Clock has not been started!"):
        self.message = message

    def __str__(self):
        return self.message


class ClockAlreadyStarted(Exception):
    def __init__(self, message="Clock has already been started!"):
        self.message = message

    def __str__(self):
        return self.message
