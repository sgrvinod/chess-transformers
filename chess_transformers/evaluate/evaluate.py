import os
import cpuinfo
import argparse

from chess_transformers.configs import import_config
from chess_transformers.play.utils import write_pgns
from chess_transformers.play import model_v_engine, warm_up, load_model, load_engine


def evaluate_model(CONFIG):
    """
    Evaluation.

    Args:

        CONFIG (dict): Configuration. See ./configs.
    """
    # Load
    model = load_model(CONFIG)
    engine = load_engine(CONFIG.FAIRY_STOCKFISH_PATH)

    # Warmup model
    warm_up(
        model=model,
    )

    # Evaluate
    for LL in range(1, 7):
        for model_color in ["w", "b"]:
            # Play
            wins, losses, draws, pgns = model_v_engine(
                model=model,
                k=CONFIG.SAMPLING_K,
                use_amp=CONFIG.USE_AMP,
                model_color=model_color,
                engine=engine,
                time_limit=CONFIG.LICHESS_LEVELS[LL]["TIME_CONSTRAINT"],
                depth_limit=CONFIG.LICHESS_LEVELS[LL]["DEPTH"],
                uci_options={
                    "Skill Level": CONFIG.LICHESS_LEVELS[LL]["SKILL"],
                    "Threads": 8,
                    "Hash": 8000,
                },
                rounds=500,
                clock=None,
                white_player_name="Fairy Stockfish @ LL {}".format(LL)
                if model_color == "b"
                else CONFIG.NAME,
                black_player_name="Fairy Stockfish @ LL {}".format(LL)
                if model_color == "w"
                else CONFIG.NAME,
                event=CONFIG.NAME + " v. Fairy Stockfish @ LL {}".format(LL)
                if model_color == "w"
                else "Fairy Stockfish @ LL {} v. ".format(LL) + CONFIG.NAME,
            )

            # Write games to PGN
            write_pgns(
                pgns,
                pgn_file=os.path.join(
                    CONFIG.EVAL_GAMES_FOLDER,
                    (
                        "LL {} | "
                        + CONFIG.NAME
                        + " as {} | GAMES {} |  W {} |  L {} |  D {} | {}.pgn"
                    ).format(
                        LL,
                        model_color.upper(),
                        500,
                        wins,
                        losses,
                        draws,
                        cpuinfo.get_cpu_info()["brand_raw"],
                    ),
                ),
            )


if __name__ == "__main__":
    # Get configuration
    parser = argparse.ArgumentParser()
    parser.add_argument("config_name", type=str, help="Name of configuration file.")
    args = parser.parse_args()
    CONFIG = import_config(args.config_name)

    # Evaluate model
    evaluate_model(CONFIG)
