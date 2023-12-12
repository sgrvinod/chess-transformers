import os
import cpuinfo

from chess_transformers.configs import import_config
from chess_transformers.play.utils import load_engine, write_pgns
from chess_transformers.play import model_v_engine, warm_up, load_model

config_name = "CT-EFT-20"
CONFIG = import_config(config_name)

model = load_model(CONFIG)

warm_up(
    model=model,
)

engine = load_engine(CONFIG.FAIRY_STOCKFISH_PATH)


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
            uci_options={"Skill Level": CONFIG.LICHESS_LEVELS[LL]["SKILL"], "Threads": 8, "Hash": 8000},
            rounds=500,
            clock=None,
            white_player_name="Fairy Stockfish @ LL {}".format(LL)
            if model_color == "b"
            else config_name,
            black_player_name="Fairy Stockfish @ LL {}".format(LL)
            if model_color == "w"
            else config_name,
            event=config_name + " v. Fairy Stockfish @ LL {}".format(LL)
            if model_color == "w"
            else "Fairy Stockfish @ LL {} v. ".format(LL) + config_name,
        )

        write_pgns(
            pgns,
            pgn_file=os.path.join(
                CONFIG.EVAL_GAMES_FOLDER,
                (
                    "LL {} | "
                    + config_name
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