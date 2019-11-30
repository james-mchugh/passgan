"""PyTorch data loader for the password dataset.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import os

# Third party imports
import numpy as np
from torch.utils.data import Dataset

# Local application imports
from passgan import utils


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

RELATIVE_DATA_PATH = "resources/data.csv"


# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class PasswordDataset(Dataset):
    def __init__(self):
        super().__init__()
        file_dir = os.path.dirname(__file__)
        self.data_path = os.path.join(file_dir, RELATIVE_DATA_PATH)
        self.data = [line for line in open(self.data_path)]

    def __getitem__(self, item: int) -> np.array:
        password = self.data[item].strip().split(':')[1]
        str_matrix = utils.vectorize_string(password, 32)
        return np.reshape(str_matrix, (1, *str_matrix.shape))

    def __len__(self):
        return len(self.data)
