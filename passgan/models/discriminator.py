"""Discrimator class for passgan.

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports

# Third party imports
from torch import nn
from torch import Tensor


# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------

# ----------------------------------------------------------------------
# class
# ----------------------------------------------------------------------

class Discriminator(nn.Module):

    def __init__(self, ngpu: int = 0):
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=(7, 1)),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding=1),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3), padding=1),
            nn.MaxPool2d(kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(1, 1, kernel_size=(1, 3)),
            nn.Sigmoid()
        )

    def forward(self, input_: Tensor):
        if input_.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input_,
                                               range(self.ngpu))
        else:
            output = self.main(input_)

        return output
