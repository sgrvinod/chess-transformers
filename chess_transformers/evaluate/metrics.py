import math
from scipy.stats import norm
from scipy.special import erfinv

"""
References:

    https://www.chessprogramming.org/Match_Statistics
    https://3dkingdoms.com/chess/elo.htm
"""


def win_ratio(wins, losses, draws):
    """
    Calculate the win ratio of a player against an opponent.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

    Returns:

        float: The win ratio.
    """

    return (wins + draws / 2) / (wins + losses + draws)


def elo_delta_from_win_ratio(wr):
    """
    Calculate difference in Elo rating between a player and their
    opponent from the player's win ratio.

    Args:

        wr (float): The win ratio.

    Returns:

        float: The difference in Elo rating.
    """
    try:
        return 400 * math.log10(wr / (1 - wr))

    except ZeroDivisionError:
        return math.inf


def elo_delta(wins, losses, draws):
    """
    Calculate difference in Elo rating between a player and their
    opponent from the player's wins, losses, and draws.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

    Returns:

        float: The difference in Elo rating.
    """
    wr = win_ratio(wins, losses, draws)

    return elo_delta_from_win_ratio(wr)


def elo_delta_margin(wins, losses, draws, confidence=0.95):
    """
    Calculate the error margin or tolerance in Elo rating difference
    between a player and their opponent, corresponding to a given
    confidence level, from the player's wins, losses, and draws.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

        draws (int): The number of draws.

        confidence (float, optional): The confidence level. Defaults to
        0.95 (95%).

    Returns:

        float: The Elo delta margin.
    """
    wr = win_ratio(wins, losses, draws)
    games = wins + losses + draws
    wins_dev = (wins / games) * math.pow(1 - wr, 2)
    draws_dev = (draws / games) * math.pow(0.5 - wr, 2)
    losses_dev = (losses / games) * math.pow(0 - wr, 2)
    std_dev = math.sqrt((wins_dev + draws_dev + losses_dev) / games)

    min_confidence = (1 - confidence) / 2
    max_confidence = 1 - min_confidence
    min_dev = wr + std_dev * norm.ppf(min_confidence)
    max_dev = min(wr + std_dev * norm.ppf(max_confidence), 0.999)

    margin = (elo_delta_from_win_ratio(max_dev) - elo_delta_from_win_ratio(min_dev)) / 2

    return margin


def likelihood_of_superiority(wins, losses):
    """
    Calculate the likelihood of superiority (LOS) of a player over their
    opponent, from their wins and losses.

    Args:

        wins (int): The number of wins for the player.

        losses (int): The number of losses for the player.

    Returns:

        float: The likelihood of superiority.
    """

    return 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * (wins + losses))))
