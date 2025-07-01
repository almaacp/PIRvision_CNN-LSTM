import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))   # Path ke direktori utama proyek

DATA_PATHS = [
    os.path.join(ROOT_DIR, "data/pirvision_office_dataset1.csv"),
    os.path.join(ROOT_DIR, "data/pirvision_office_dataset2.csv"),
]

RANDOM_SEED = 42    # Seed untuk random state agar hasil reprodusibel
TEST_SIZE = 0.15    # Ukuran split untuk data testing
VAL_SIZE = 0.15 # Ukuran split untuk data validation
WINDOW_SIZE = 8 # Ukuran windows untuk segmentasi data