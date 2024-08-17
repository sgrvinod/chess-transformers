# Change Log

## v0.3.0

### Added

* There are 3 new datasets: [ML23c](https://github.com/sgrvinod/chess-transformers#ml23c), [GC22c](https://github.com/sgrvinod/chess-transformers#gc22c), and [ML23d](https://github.com/sgrvinod/chess-transformers#ml23d).
* A new naming convention for datasets is used. Datasets are now named in the format "[*PGN Fileset*][*Filters*]". For example, *LE1222* is now called [*LE22ct*](https://github.com/sgrvinod/chess-transformers#le22ct), where *LE22* is the name of the PGN fileset from which this dataset was derived, and "*c*", "*t*" are filters for games that ended in checkmates and games that used a specific time control respectively.
* [*CT-EFT-85*](https://github.com/sgrvinod/chess-transformers#ct-eft-85) is a new trained model with about 85 million parameters.
* **`chess_transformers.train.utils.get_lr()`** now accepts new arguments, `schedule` and `decay`, to accomodate a new learning rate schedule: exponential decay after warmup.
* **`chess_transformers.data.prepare_data()`** now handles errors where there is a mismatch between the number of moves and the number of FENs, or when the recorded result in the PGN file was incorrect. Such games are now reported and excluded during dataset creation.

### Changed

* The *LE1222* and *LE1222x* datasets are now renamed to [*LE22ct*](https://github.com/sgrvinod/chess-transformers#le22ct) and [*LE22c*](https://github.com/sgrvinod/chess-transformers#le22c) respectively.
* All calls to **`chess_transformers.train.utils.get_lr()`** now use the `schedule` and `decay` arguments, even in cases where a user-defined decay is not required.
* **`chess_transformers.train.datasets.ChessDatasetFT`** was optimized for large datasets. A list of indices for the data split is no longer maintained or indexed in the dataset.
* Dependencies in [**`setup.py`**](https://github.com/sgrvinod/chess-transformers/blob/main/setup.py) have been updated to newer versions.
* Fixed an error in **`chess_transformers.play.model_v_model()`** where a move would be attempted by the model playing black even after white won the game with a checkmate.
* Fixed the `EVAL_GAMES_FOLDER` parameter in the model configuration files pointing to the incorrect folder name **`chess_transformers/eval`** instead of **`chess_transformers/evaluate`**.
* Fixed an error in **`chess_transformers.evaluate.metrics.elo_delta_margin()`** where the upper limit of the win ratio for the confidence interval was not capped at a value of 100%.
* All calls to `torch.load()` now use `weights_only=True` in compliance with its updated API.

## v0.2.1

### Changed

* All model checkpoints, datasets, and logs have now been moved to Microsoft Azure Storage for more reliable access.

## v0.2.0

### Added

* **`ChessTransformerEncoderFT`** is an encoder-only transformer that predicts source (*From*) and destination squares (*To*) squares for the next half-move, instead of the half-move in UCI notation.
* [*CT-EFT-20*](https://github.com/sgrvinod/chess-transformers#ct-eft-20) is a new trained model of this type with about 20 million parameters.
* **`ChessDatasetFT`** is a PyTorch dataset class for this model type.
* [**`chess_transformers.data.levels`**](https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/data/levels.py) provides a standardized vocabulary (with indices) for oft-used categorical variables. All models and datasets will hereon use this standard vocabulary instead of a dataset-specific vocabulary.

### Changed

* The [*LE1222*](https://github.com/sgrvinod/chess-transformers#le1222) and [*LE1222x*](https://github.com/sgrvinod/chess-transformers#le1222x) datasets no longer have their own vocabularies or vocabulary files. Instead, they use the standard vocabulary from **`chess_transformer.data.levels`**.
* The [*LE1222*](https://github.com/sgrvinod/chess-transformers#le1222) and [*LE1222x*](https://github.com/sgrvinod/chess-transformers#le1222x) datasets have been re-encoded with indices corresponding to the standard vocabulary. Earlier versions or downloads of these datasets are no longer valid for use with this library.
* The row index at which the validation split begins in each dataset is now stored as an attribute of the **`encoded_data`** table in the corresponding H5 file, instead of in a separate JSON file.
* Models [*CT-E-20*](https://github.com/sgrvinod/chess-transformers#ct-e-20) and [*CT-ED-45*](https://github.com/sgrvinod/chess-transformers#ct-ed-45) already trained with a non-standard, dataset-specific vocabulary have been refactored for use with the standard vocabulary. Earlier versions or downloads of these models are no longer valid for use with this library.
* The field **`move_sequence`** in the H5 tables has now been renamed to **`moves`**.
* The field **`move_sequence_length`** in the H5 tables has now been renamed to **`length`**.
* The **`load_assets()`** function has been renamed to **`load_model()`** and it no longer returns a vocabulary — only the model.
* The **`chess_transformers/eval`** folder has been renamed to [**`chess_transformers/evaluate`**](https://github.com/sgrvinod/chess-transformers/tree/main/chess_transformers/evaluate). 
* The Python notebook **`lichess_eval.ipynb`** has been converted to a Python script [**`evaluate.py`**](https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/evaluate/evaluation.py), which runs much faster for evaluation. 
* Fairy Stockfish is now run on 8 threads and with a hash table of size 8 GB during evaluation instead of 1 thread and 16 MB respectively, which makes it a more challenging opponent.
* Evaluation results have been recomputed for [*CT-E-20*](https://github.com/sgrvinod/chess-transformers#ct-e-20) and [*CT-ED-45*](https://github.com/sgrvinod/chess-transformers#ct-ed-45) against this stronger Fairy Stockfish — they naturally fare worse.

### Removed

* The environment variable **`CT_LOGS_FOLDER`** no longer needs to be set before training a model. Training logs will now always be saved to **`chess_transformers/training/logs`**. 
* The environment variable **`CT_CHECKPOINT_FOLDER`** no longer needs to be set before training a model. Checkpoints will now always be saved to **`chess_transformers/checkpoints`**.
* The environment variable **`CT_EVAL_GAMES_FOLDER`** no longer needs to be set before evaluating a model. Evaluation games will now always be saved to **`chess_transformers/evaluate/games`**.

  
