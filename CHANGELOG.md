# Change Log

## v0.2.1

### Changed

* All model checkpoints, datasets, and logs have now been moved to Microsoft Azure Storage for more reliable access.

## v0.2.0

### Added

* **`ChessTransformerEncoderFT`** is an encoder-only transformer that predicts source (*From*) and destination squares (*To*) squares for the next half-move, instead of the half-move in UCI notation.
* [*CT-EFT-20*](https://github.com/sgrvinod/chess-transformers#ct-eft-20) is a new trained model of this type with about 20 million parameters.
* **`ChessDatasetFT`** is a PyTorch dataset class for this model type.
* [**`chess_transformer.data.levels`**](https://github.com/sgrvinod/chess-transformers/blob/main/chess_transformers/data/levels.py) provides a standardized vocabulary (with indices) for oft-used categorical variables. All models and datasets will hereon use this standard vocabulary instead of a dataset-specific vocabulary.

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

  
