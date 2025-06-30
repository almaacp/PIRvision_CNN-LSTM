import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

DATA_PATHS = [
    os.path.join(ROOT_DIR, "data/pirvision_office_dataset1.csv"),
    os.path.join(ROOT_DIR, "data/pirvision_office_dataset2.csv"),
]

RANDOM_SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
WINDOW_SIZE = 8