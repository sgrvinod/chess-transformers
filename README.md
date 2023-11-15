<p align="center">
  <img width="300" src="img/logo.png"/>
</p>

<h1 align="center"><i>Chess Transformers</i></h1>
<p align="center"><i>Teaching transformers to play chess</i></p>
<br>

*Chess Transformers* contains code for training transformer models to play chess by learning from human games. 

This is in initial development and the public API may change at any time — maybe even thanks to [*you*]()! 

## Contents

[**Install**]()

[**Models**]()

[**Datasets**]()

[**Play**]()

[**Train**]()

[**Contribute**]()


## Install

To install *Chess Transformers*, clone this repository and install as a Python package locally.

```
gh repo clone sgrvinod/chess-transformers
cd chess-transformers
pip install -e .
```

If you are planning to develop or contribute to the codebase, install the package in <ins>editable mode</ins>, using the `-e` flag.

```
pip install -e .
```

Additionally, <ins>set the following environment variables</ins> on your computer in any manner of your choosing.

- <ins>Required</ins> environment variables:

  - Set **`CT_DATA_FOLDER`** to the folder on your computer where you have the training data, which also includes the vocabulary file needed for any inference with the model of your choice. You will need to [download]() the training data manually. Each vocabulary file will need to be *in its own folder* in **`CT_DATA_FOLDER`** with the name of the folder being the name of the dataset. For example, the vocabulary file **`vocabulary.json`** for [*LE1222*]() should be located in the folder **`[$CT_DATA_FOLDER]/LE1222/`**.

  - Set **`CT_CHECKPOINTS_FOLDER`** to the folder on your computer where you have the model checkpoints. You will need to download model checkpoints manually. Each checkpoint will need to be *in its own folder* in **`CT_CHECKPOINTS_FOLDER`** with the name of the folder being the name of the model. For example, the checkpoint **`averaged_CT-E-20.pt`** for [*CT-E-20*]() should be located in the folder **`[$CT_CHECKPOINTS_FOLDER]/CT-E-20/`**.

- <ins>Optional</ins> environment variables:
    
  - Set **`CT_LOGS_FOLDER`** to the folder on your computer where you wish to save the training logs. You do <ins>not</ins> need to set this if you do not plan to train any models.
    
  - Set **`CT_STOCKFISH_PATH`** to the executable of the Stockfish 16 chess engine. You do <ins>not</ins> need to set this if you do not plan to have a model play against this chess engine.

  - Set **`CT_FAIRY_STOCKFISH_PATH`** to the executable of the Fairy Stockfish chess engine. You do <ins>not</ins> need to set this if you do not plan to have a model play against this chess engine.

  - Set **`CT_EVAL_GAMES_FOLDER`** to the folder where you want to save PGN files for evaluation games. You do <ins>not</ins> need to set this if you do not plan to evaluate any model.

## Models

There are currently two chess-transformer models available for use.

|     Model Name     | # Params |  Training Data   |            Architecture             |                                                                      Predictions                                                                      |
| :----------------: | :------: | :--------------: | :---------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------: |
| [***CT-E-20***]()  |   20M    | [***LE1222***]() |      Transformer encoder only       |                                                     Best next half-move (or ply) <br> eg. *f2e3*                                                      |
| [***CT-ED-45***]() |   45M    | [***LE1222***]() | Transformer encoder <br>and decoder | Sequence of half-moves (or plies) <br> eg. *f2e3* -> *b4b3* -> *e3h6* -> *b3b2* -> *g4e6* -> *g8f8* -> *g3g7* -> *f8e8* -> *g7f7* -> *loses* |

### *CT-E-20*

[**Configuration File**]() | [**Download Checkpoint**]() | 
[**Download TensorBoard Logs**]() 

This is the encoder from the original transformer model in [*Vaswani et al. (2017)*](https://arxiv.org/abs/1706.03762) trained on the [*LE1222*]() dataset. A classification head at the **`turn`** token predicts the best half-move to be made (in UCI notation) on the board in its current state. 

This is essentially a sequence (or image) classification task, where the sequence is the current state of the board, and the classes are the various moves that can be made on a chessboard in UCI notation. 

<p align="center">
  <img src="img/ct_e_20.png"/>
</p>

*CT-E-20* contains about <ins>20 million parameters</ins>.

```python
from chess_transformers.configs import import_config

CONFIG = import_config("CT-E-20")
model, vocabulary = load_assets(CONFIG)
```

#### Model Skill

*CT-E-20* model was evaluated against the Fairy Stockfish chess engine at various skill levels [as predefined](https://github.com/lichess-org/fishnet/blob/dc4be23256e3e5591578f0901f98f5835a138d73/src/api.rs#L224) for use in the popular Stockfish chess bots on Lichess, with the engine running on an AMD Ryzen 7 3800X 8-Core Processor.

These evaluation games can be viewed [here]().

| Lichess Level | Games | Wins  | Losses | Draws |          Win Ratio          |      ELO Difference      | Likelihood of Superiority |
| :-----------: | :---: | :---: | :----: | :---: | :-------------------------: | :----------------------: | :-----------------------: |
|     $LL$      |  $n$  |  $w$  |  $l$   |  $d$  | $\frac{w + \frac{d}{2}}{n}$ |      $\Delta_{ELO}$      |           $LOS$           |
|     **1**     | 1000  |  987  |   0    |  13   |         **99.35%**          | 873.70 <br> *(± 105.58)* |          100.00%          |
|     **2**     | 1000  |  984  |   0    |  16   |         **99.20%**          | 837.37 <br> *(± 92.90)*  |          100.00%          |
|     **3**     | 1000  |  854  |   72   |  74   |         **89.10%**          | 364.98 <br> *(± 31.32)*  |          100.00%          |
|     **4**     | 1000  |  551  |  330   |  119  |         **61.05%**          |  78.07 <br> *(± 20.68)*  |          100.00%          |
|     **5**     | 1000  |  361  |  503   |  136  |         **42.90%**          | -49.67 <br> *(± 20.21)*  |           0.00%           |
|     **6**     | 1000  |  53   |  897   |  50   |          **7.80%**          | -429.05 <br> *(± 36.92)* |           0.00%           |

### *CT-ED-45*

[**Configuration File**]() | [**Download Checkpoint**]() | 
[**Download TensorBoard Logs**]() 

This is the original transformer model (encoder *and* decoder) in [*Vaswani et al. (2017)*](https://arxiv.org/abs/1706.03762) trained on the [*LE1222*]() dataset. A classification head after the last decoder layer predicts a sequence of half-moves, starting with the best half-move to be made on the board in its current state, followed by the likely course of the game an arbitrary number of half-moves into the future. 

This is essentially a sequence-to-sequence (or image-to-sequence) task, where the input sequence is the current state of the board, and the output sequence is the string of half-moves that will likely occur on the board from that point onwards. Potentially, strategies applied to such tasks, such as beam search for decoding the best possible sequence of half-moves, can also be applied. 
Training the model to predict not only the best half-move to make on the board right now but also the sequence of half-moves that follow can be viewed as a type of multitask training, although we are ultimately only interested in the very first half-move. This sequence of half-moves might also lend itself to making the model more "explainable" regarding its decision for that important first half-move.

<p align="center">
  <img src="img/ct_ed_45.png"/>
</p>

*CT-ED-45* contains about <ins>45 million parameters</ins>.

```python
from chess_transformers.configs import import_config

CONFIG = import_config("CT-ED-45")
model, vocabulary = load_assets(CONFIG)
```

#### Model Skill

*CT-ED-45* was evaluated against the Fairy Stockfish chess engine at various skill levels [as predefined](https://github.com/lichess-org/fishnet/blob/dc4be23256e3e5591578f0901f98f5835a138d73/src/api.rs#L224) for use in the popular Stockfish chess bots on Lichess, with the engine running on an AMD Ryzen 7 3800X 8-Core Processor.

| Lichess Level | Games | Wins  | Losses | Draws |          Win Ratio          |      ELO Difference      | Likelihood of Superiority |
| :-----------: | :---: | :---: | :----: | :---: | :-------------------------: | :----------------------: | :-----------------------: |
|     $LL$      |  $n$  |  $w$  |  $l$   |  $d$  | $\frac{w + \frac{d}{2}}{n}$ |      $\Delta_{ELO}$      |           $LOS$           |
|     **1**     | 1000  |  991  |   0    |   9   |         **99.55%**          | 937.93 <br> *(± 135.31)* |          100.00%          |
|     **2**     | 1000  |  976  |   0    |  24   |         **98.80%**          | 766.23 <br> *(± 73.45)*  |          100.00%          |
|     **3**     | 1000  |  695  |  214   |  91   |         **74.05%**          | 182.16 <br> *(± 23.12)*  |          100.00%          |
|     **4**     | 1000  |  251  |  666   |  83   |         **29.25%**          | -153.44 <br> *(± 22.50)* |           0.00%           |
|     **5**     | 1000  |  130  |  790   |  80   |         **17.00%**          | -275.45 <br> *(± 26.67)* |           0.00%           |
|     **6**     | 1000  |   6   |  969   |  25   |          **1.85%**          | -689.89 <br> *(± 67.79)* |           0.00%           |

## Datasets

There are currently two training datasets.

|   Dataset Name    |                                    Components                                    | # Datapoints |
| :---------------: | :------------------------------------------------------------------------------: | :----------: |
| [***LE1222***]()  | Board positions, turn, castling rights, next-move sequence (up to 10 half-moves) |  13,287,522  |
| [***LE1222x***]() | Board positions, turn, castling rights, next-move sequence (up to 10 half-moves) | 127,684,720  |

### *LE1222*

This consists of games from the [Lichess Elite Database](https://database.nikonoel.fr/) put together by [nikonoel](https://lichess.org/@/nikonoel), a collection of all standard chess games played on [Lichess.org]() by players with a Lichess Elo rating of 2400+ against players with a Lichess Elo rating of 2200+ up to December 2021, and players rated 2500+ against players rated 2300+ from December 2021. 

On this data, we apply the following filters to keep only those games that:

- were played up to December 2022 (20,241,368 games)
- and used a time control of at least 5 minutes (2,073,780 games)
- and ended in a checkmate (**274,794 games**)

These 274,794 games consist of a total **13,287,522 half-moves (or plies)** made by the <ins>winners</ins> of the games, which alone constitute the dataset. For each such half-move, the board positions (or layout), turn (white or black), and castling rights of both players before the move are calculated, as well as the sequence of half-moves beginning with this half-move up to 10 half-moves into the future. Draw potential is not calculated.

[**Download here.**]() The data is zipped and will need to be extracted.

It consists of the following files:

- **`LE1222.h5`**, an HDF5 file containing two tables, one with the raw data and the other encoded with indices (that will be used in the transformer model), containing the following fields:
  - **`board_position`**, the board layout or positions of pieces on the board
  - **`turn`**, the color of the pieces of the player to play
  - **`white_kingside_castling_rights`**, whether white can castle kingside
  - **`white_queenside_castling_rights`**, whether white can castle queenside
  - **`black_kingside_castling_rights`**, whether black can castle kingside
  - **`black_queenside_castling_rights`**, whether black can castle queenside
  - **`move_sequence`**, 10 half-moves into the future made by both players
  - **`move_sequence_length`**, the number of half-moves in the sequence, as this will be less than 10 at the end of the game
- **`vocabulary.json`**, consisting of mappings between raw data and the integers they are encoded as in the HDF5 file, for all variables
- **`splits.json`**, consisting of the index in the tables at which the dataset is split into training and validation data

You can ignore any file that is named **`#.fens`** or **`#.moves`** — these are intermediate files from the dataset creation process.

### *LE1222x*

This is an extended version of [*LE1222*](), and also consists of games from the [Lichess Elite Database](https://database.nikonoel.fr/) put together by [nikonoel](https://lichess.org/@/nikonoel), a collection of all standard chess games played on [Lichess.org]() by players with a Lichess Elo rating of 2400+ against players with a Lichess Elo rating of 2200+ up to December 2021, and players rated 2500+ against players rated 2300+ from December 2021. 

On this data, we apply the following filters to keep only those games that:

- were played up to December 2022 (20,241,368 games)
- and ended in a checkmate (**2,751,394 games**)

These 2,751,394 games consist of a total **127,684,720 half-moves (or plies)** made by the <ins>winners</ins> of the games, which alone constitute the dataset. For each such half-move, the board positions (or layout), turn (white or black), and castling rights of both players before the move are calculated, as well as the sequence of half-moves beginning with this half-move up to 10 half-moves into the future. Draw potential is not calculated.

[**Download here.**]() The data is zipped and will need to be extracted.

It consists of the following files:

- **`LE1222.h5`**, an HDF5 file containing two tables, one with the raw data and the other encoded with indices (that will be used in the transformer model), containing the following fields:
  - **`board_position`**, the board layout or positions of pieces on the board
  - **`turn`**, the color of the pieces of the player to play
  - **`white_kingside_castling_rights`**, whether white can castle kingside
  - **`white_queenside_castling_rights`**, whether white can castle queenside
  - **`black_kingside_castling_rights`**, whether black can castle kingside
  - **`black_queenside_castling_rights`**, whether black can castle queenside
  - **`move_sequence`**, 10 half-moves into the future made by both players
  - **`move_sequence_length`**, the number of half-moves in the sequence, as this will be less than 10 at the end of the game
- **`vocabulary.json`**, consisting of mappings between raw data and the integers they are encoded as in the HDF5 file, for all variables
- **`splits.json`**, consisting of the index in the tables at which the dataset is split into training and validation data

You can ignore any file that is named **`#.fens`** or **`#.moves`** — these are intermediate files from the dataset creation process.

## Play

After [installing]() *Chess Transformers*, you can play games <ins>against an available model</ins> or have a model play <ins>against a chess engine</ins>.

### You v. Model

You could either play in a Jupyter notebook (recommended for better UI) or in the regular Python shell. 

```python
import os
from chess_transformers.configs import import_config
from chess_transformers.play import human_v_model, warm_up
from chess_transformers.play.utils import load_assets, write_pgns

# Load configuration
config_name = "CT-E-20"
CONFIG = import_config(config_name)

# Load assets
model, vocabulary = load_assets(CONFIG)

# Warmup model (triggers compilation)
warm_up(
    model=model,
    vocabulary=vocabulary,
)

# Play
wins, laws, draws, pgns = human_v_model(
    human_color="b",  # color you want to play
    model=model,
    vocabulary=vocabulary,
    k=1,  # "k" in "top_k sampling", k=1 is best
    use_amp=True,
    rounds=1,  # number of rounds you want to play
    clock=None, 
    white_player_name=config_name,
    black_player_name="Me",
)

# Print games in Portable Game Notation (PGN) format
print(pgns)

# Save PGNs if you wish
write_pgns(
    pgns,
    pgn_file="somewhere/something.pgn",
)
```

You could also just make a copy of [**`human_play.ipynb`**]() and play in that notebook.

### Model v. Engine

The process is the same as above, except you must use a different set of functions:

```python
from chess_transformers.play import model_v_engine
from chess_transformers.play.utils import load_engine

engine = load_engine(CONFIG.FAIRY_STOCKFISH_PATH)

LL = 1  # Try Lichess levels 1 to 8 (note: 7 and 8 may be slow)
model_color = "w"  # Try "w" and "b"

# Play
wins, losses, draws, pgns = model_v_engine(
    model=model,
    vocabulary=vocabulary,
    k=CONFIG.SAMPLING_K,
    use_amp=CONFIG.USE_AMP,
    model_color=model_color,
    engine=engine,
    time_limit=CONFIG.LICHESS_LEVELS[LL]["TIME_CONSTRAINT"],
    depth_limit=CONFIG.LICHESS_LEVELS[LL]["DEPTH"],
    uci_options={"Skill Level": CONFIG.LICHESS_LEVELS[LL]["SKILL"]},
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
```
See [**`lichess_eval.ipynb`**]() for an example.

### Time Control

If you're using a *Unix*-type operating system — basically, not Windows — you can also set time control for your games. 
Currently, only Fischer time control is available. 

```python
from chess_transformers.play.clocks import ChessClock

clock = ChessClock(base_time=60, 
                   increment=1)
```

Pass this **`clock`** to the functions above instead of **`clock=None`**.

## Train Models

You're welcome to try to train your own models, but if you wish to contribute trained models, please [discuss first]().

### Dataset

You can skip this step if you wish to use one of the [existing datasets]().

- Collect PGN files of games you wish to use for training the model.

- Create a bash script for parsing these PGN files into a collection of FENs and moves using [*pgn-extract*](), like in [**`LE1222.sh`**]().

- Create a configuration file for the dataset, like in [**`LE1222.py`**]().

- Run [**`prep.py`**]() like `python prep.py [config_name]` or do it in your own Python script.

```python
from chess_transformers.data import prepare_data
from chess_transformers.configs import import_config

# Load configuration
CONFIG = import_config("[config_name]")

# Prepare data
prepare_data(
    data_folder=CONFIG.DATA_FOLDER,
    h5_file=CONFIG.H5_FILE,
    max_move_sequence_length=CONFIG.MAX_MOVE_SEQUENCE_LENGTH,
    expected_rows=CONFIG.EXPECTED_ROWS,
    vocab_file=CONFIG.VOCAB_FILE,
    splits_file=CONFIG.SPLITS_FILE,
    val_split_fraction=CONFIG.VAL_SPLIT_FRACTION,
)
```
This will create all data files in **`CONFIG.DATA_FOLDER`**.

### Training

- Create a configuration file for the model, like in [**`CT-E-20.py`**]().

- Run [**`train.py`**]() like `python train.py [config_name]` or do it in your own Python script.

```python

from chess_transformers.train import train_model
from chess_transformers.data.prep import prepare_data

# Load configuration
CONFIG = import_config("[config_name]")

# Train model
train_model(CONFIG)
```
- Monitor training with [*TensorBoard*]() with `tensorboard --logdir $CT_LOGS_DIR`.

### Evaluation

Evaluate in [**`lichess_eval.ipynb`**]() or use that code in your own Python notebook/script.

## Contribute

Contributions — and any discussion thereof — are welcome. As you may have noticed, *Chess Transformers* is in initial development and the public API is <ins>not</ins> to be considered stable. 

If you are planning to contribute bug-fixes, please go ahead and do so. If you are planning to contribute in a way that extends *Chess Transformers* or adds any new features, data, or models, please [open an issue]() to discuss it <ins>before</ins> you spend any time on it. Otherwise, your PR may be rejected due to lack of consensus or alignment with current goals.

Presently, the following types of contributions may be useful:

- Better, more robust evaluation methods of chess-transformer models.
- Evaluation of existing models against chess engines on different CPUs to study the effect of CPU specifications on engine strength and evaluation.
- New models with:
  - same transformer architecture but larger size, and on larger datasets.
  - or different transformer architectures or internal mechanisms.
  - or in general, improved evaluation scores.
- Chess clocks for Windows OS, or for *Unix*-type OS but for time controls <ins>other than</ins> Fischer time control.
- Refactoring of code that improves its ease of use.
- Model visualization for explainable AI, such as visualizing positional or move embeddings, or attention patterns.
- Streamline installation of this package, such as with automatic downloading of assets (training data, vocabularies, model checkpoints, etc.)

This list is not exhaustive. Please do not hesitate to discuss your ideas. Thank you!

## License

*Chess Transformers* is licensed under the [MIT license](). 






