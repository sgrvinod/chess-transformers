import math
from scipy.stats import norm
from scipy.special import erfinv


def win_ratio(wins, losses, draws):
    return (wins + draws / 2) / (wins + losses + draws)


def elo_delta_from_win_ratio(wr):
    try:
        return 400 * math.log10(wr / (1 - wr))
    except ZeroDivisionError:
        return math.inf


def elo_delta(wins, losses, draws):
    wr = win_ratio(wins, losses, draws)
    return elo_delta_from_win_ratio(wr)


def elo_delta_margin(wins, losses, draws):
    wr = win_ratio(wins, losses, draws)
    games = wins + losses + draws
    wins_dev = (wins / games) * math.pow(1 - wr, 2)
    draws_dev = (draws / games) * math.pow(0.5 - wr, 2)
    losses_dev = (losses / games) * math.pow(0 - wr, 2)
    std_dev = math.sqrt((wins_dev + draws_dev + losses_dev) / games)

    confidence = 0.95
    min_confidence = (1 - confidence) / 2
    max_confidence = 1 - min_confidence
    min_dev = wr + std_dev * norm.ppf(min_confidence)
    max_dev = wr + std_dev * norm.ppf(max_confidence)

    margin = (elo_delta_from_win_ratio(max_dev) - elo_delta_from_win_ratio(min_dev)) / 2

    return margin


def likelihood_of_superiority(wins, losses):
    return 0.5 * (1 + math.erf((wins - losses) / math.sqrt(2 * (wins + losses))))
