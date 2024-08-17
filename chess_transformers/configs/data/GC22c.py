import os

###############################
############ Name #############
###############################

NAME = "GC22c"  # name and identifier for this configuration

###############################
############ Data #############
###############################

DATA_FOLDER = (
    os.path.join(os.environ.get("CT_DATA_FOLDER"), NAME)
    if os.environ.get("CT_DATA_FOLDER")
    else None
)  # folder containing all data files
H5_FILE = NAME + ".h5"  # H5 file containing data
MAX_MOVE_SEQUENCE_LENGTH = 10  # expected maximum length of move sequences
EXPECTED_ROWS = 27000000  # expected number of rows, approximately, in the data
VAL_SPLIT_FRACTION = 0.95  # marker (% into the data) where the validation split begins
