"""Main script for running PassGan

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import argparse
import random

# Third party imports
import torch

# Local application imports
from passgan.models.discriminator import Discriminator
from passgan import utils

# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train PassGan model.")
    parser.add_argument("--batch_size", "b", type=int, default=32,
                        help="Number of samples per batch.")
    parser.add_argument("--epochs", "e", type=int, default=2000,
                        help="Number of training epochs.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of dataloader workers to use.")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="Number of GPUs to use.")
    parser.add_argument("--seed", "-s", type=int,
                        default=None,
                        help="Random seed for reproducible experiments.")
    args = parser.parse_args()

    seed = args.seed if args.seed else random.randint(1, 200000)
    torch.manual_seed = seed

    device = torch.device("cuda:0") if args.ngpu > 0 else "cpu"

    dnet = Discriminator(args.ngpu).to(device)
    dnet.zero_grad()
    dnet.apply(utils.init_weights)




if __name__ == '__main__':
    main()
