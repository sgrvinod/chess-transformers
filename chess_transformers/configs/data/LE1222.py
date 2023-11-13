import os

###############################
############ Name #############
###############################

NAME = "LE1222"  # name and identifier for this configuration

###############################
############ Data #############
###############################

DATA_FOLDER = os.path.join(
    os.environ["CT_DATA_FOLDER"], NAME
)  # folder containing all data files
H5_FILE = NAME + ".h5"  # H5 file containing data
MAX_MOVE_SEQUENCE_LENGTH = 10  # expected maximum length of move sequences
EXPECTED_ROWS = 12500000  # expected number of rows, approximately, in the data
SPLITS_FILE = "splits.json"  # splits file
VOCAB_FILE = "vocabulary.json"  # vocabulary file
VAL_SPLIT_FRACTION = 0.925  # marker (% into the data) where the validation split begins
