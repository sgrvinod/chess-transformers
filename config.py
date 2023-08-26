import torch
from utils import *
from torch.utils.tensorboard import SummaryWriter

# Device
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)  # CPU isn't really practical here

# Data
DATA_FOLDER = "/media/sgr/SSD/lichess data (copy)/"  # folder containing all data files
H5_FILE = "data.h5"  # H5 file containing data
SPLITS_FILE = "splits.json"  # splits file
VOCAB_FILE = "vocabulary.json"  # vocabulary file
MOVE_SEQUENCE_NOTATION = "uci"  # chess notation format for moves
VAL_SPLIT_FRACTION = 0.85  # marker (% into the data) where the validation split begins
TEST_SPLIT_FRACTION = (
    0.925  # marker (% into the data) where the validation split ends / test set begins
)

# Checkpoint
CHECKPOINT_FOLDER = "./"  # folder containing checkpoints
TRAINING_CHECKPOINT = None  # path to model checkpoint to resume training, None if none
CHECKPOINT_AVG_PREFIX = (
    "step"  # prefix to add to checkpoint name when saving checkpoints for averaging
)
CHECKPOINT_AVG_SUFFIX = (
    ".pt"  # checkpoint end string to match checkpoints saved for averaging
)
FINAL_CHECKPOINT = "averaged_transformer_checkpoint.pt"  # final checkpoint to be used for eval/inference

# Model
D_MODEL = 512  # size of vectors throughout the transformer model
N_HEADS = 8  # number of heads in the multi-head attention
D_QUERIES = 64  # size of query vectors (and also the size of the key vectors) in the multi-head attention
D_VALUES = 64  # size of value vectors in the multi-head attention
D_INNER = 2048  # an intermediate size in the position-wise FC
N_LAYERS = 6  # number of layers in the Encoder and Decoder
DROPOUT = 0.1  # dropout probability
MAX_MOVE_SEQUENCE_LENGTH = 10  # expected maximum length of move sequences
COMPILE_MODE = "default"  # mode of model compilation (see torch.compile())
DYNAMIC_COMPILE = True  # expect tensors with dynamic shapes?
SAMPLING_K = 1  # k in top-k sampling model predictions during play

# Learning
BATCH_SIZE = 512  # batch size
BATCHES_PER_STEP = (
    2048 // BATCH_SIZE
)  # perform a training step, i.e. update parameters, once every so many batches
PRINT_FREQUENCY = 1  # print status once every so many steps
N_STEPS = 100000  # number of training steps
WARMUP_STEPS = 8000  # number of warmup steps where learning rate is increased linearly; twice the value in the paper, as in the official transformer repo.
STEP = 1  # the step number, start from 1 to prevent math error in the next line
LR = get_lr(
    step=STEP, d_model=D_MODEL, warmup_steps=WARMUP_STEPS
)  # see utils.py for learning rate schedule; twice the schedule in the paper, as in the official transformer repo.
START_EPOCH = 0  # start at this epoch
BETAS = (0.9, 0.98)  # beta coefficients in the Adam optimizer
EPSILON = 1e-9  # epsilon term in the Adam optimizer
LABEL_SMOOTHING = 0.1  # label smoothing co-efficient in the Cross Entropy loss
NUM_WORKERS = 8  # number of workers to use for dataloading
PREFETCH_FACTOR = 2  # number of batches to prefetch per worker
PIN_MEMORY = False  # pin to GPU memory when dataloading?
BOARD_STATUS_LENGTH = 70  # total length of input sequence
LOG_DIR = "runs/vanilla"  # folder for tensorboard training logs
WRITER = SummaryWriter(log_dir=LOG_DIR)  # tensorboard writer
USE_AMP = True  # use automatic mixed precision training?

# Stockfish
STOCKFISH_PATH = (
    "/home/sgr/projects/stockfish/src/stockfish"  # path to Stockfish engine
)
FAIRY_STOCKFISH_PATH = "/home/sgr/projects/fairy-stockfish/fairy-stockfish-largeboard_x86-64"  # path to Fairy Stockfish engine
LICHESS_LEVELS = {
    1: {"SKILL": -9, "DEPTH": 5, "TIME_CONSTRAINT": 50},
    2: {"SKILL": -5, "DEPTH": 5, "TIME_CONSTRAINT": 100},
    3: {"SKILL": -1, "DEPTH": 5, "TIME_CONSTRAINT": 150},
    4: {"SKILL": 3, "DEPTH": 5, "TIME_CONSTRAINT": 200},
    5: {"SKILL": 7, "DEPTH": 5, "TIME_CONSTRAINT": 300},
    6: {"SKILL": 11, "DEPTH": 8, "TIME_CONSTRAINT": 400},
    7: {"SKILL": 16, "DEPTH": 13, "TIME_CONSTRAINT": 500},
    8: {"SKILL": 20, "DEPTH": 22, "TIME_CONSTRAINT": 1000},
}  # from https://github.com/lichess-org/fishnet/blob/dc4be23256e3e5591578f0901f98f5835a138d73/src/api.rs#L224
PGN_FOLDER = (
    "./stockfish_play"  # folder where games against Stockfish are saved in PGN files
)
PGN_FILE = "LL {} | CT {} | GAMES {} |  W {} |  L {} |  D {}.pgn"  # format for PGN files' names
