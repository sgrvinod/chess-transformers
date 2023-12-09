# Change Log

## Unreleased (v0.2.0)

### Added

* **`ChessTransformerEncoderFT`** is an encoder-only transformer that predicts "From" and "To" squares for the next move, instead of the move in UCI notation.
* [*CT-EFT-20*]() is a new trained model of this type with about 20 million parameters.
* **`ChessDatasetFT`** is a PyTorch dataset class for this model type.
* **`chess_transformer.data.levels`** is a standardized vocabulary (with indices) set for oft-used categorical variables.

### Changed

* All future models now use a standardized vocabulary instead of dataset-specific vocabulary. 
* [*LE1222*]() and [*LE1222x*]() datasets no longer have their own vocabularies or vocabulary files. Instead, they use the standard vocabulary from **`chess_transformer.data.levels`**.
* The row index at which the validation split begins in each dataset will now be stored as an attribute of the **`encoded_data`** table in the corresponding H5 file..
* Split files are longer be used for any dataset. 
* Models [*CT-E-20*]() and [*CT-ED-45*]() already trained with a non-standard, dataset-specific vocabulary must continue to be used with that dataset-specific vocabulary. The dataset [*LE1222*]() used to train these models, with data encoded using the dataset-specific vocabulary, is now designated [*LE1222-legacy*]().
* The field **`move_sequence`** in the H5 tables has now been renamed to **`moves`**.
* The field **`move_sequence_length`** in the H5 tables has now been renamed to **`length`**.
  
