#!/usr/bin/env python3
"""Main script for running PassGan

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import argparse
import random
import sys

# Third party imports
import torch
from torch.utils.data import DataLoader

# Local application imports
from passgan.datasets.linkedin import PasswordDataset
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
    parser.add_argument("--batch_size", "-b", type=int, default=32,
                        help="Number of samples per batch.")
    parser.add_argument("--epochs", "-e", type=int, default=2000,
                        help="Number of training epochs.")
    parser.add_argument("--lr", "-r", type=float, default=0.01,
                        help="Learning rate for training.")
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of dataloader workers to use.")
    parser.add_argument("--ngpu", type=int, default=0,
                        help="Number of GPUs to use.")
    parser.add_argument("--seed", "-s", type=int,
                        default=None,
                        help="Random seed for reproducible experiments.")
    args = parser.parse_args()

    seed = args.seed if args.seed else random.randint(1, 200000)
    torch.manual_seed(seed)

    dataloader = DataLoader(PasswordDataset(), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=True)

    device = torch.device("cuda:0") if args.ngpu > 0 else "cpu"

    dnet = Discriminator(args.ngpu).to(device)
    dnet.apply(utils.init_weights)
    optimizer = torch.optim.Adam(dnet.parameters(), lr=args.lr)

    scorer = torch.nn.MSELoss()

    num_batches = len(dataloader)

    for n in range(args.epochs):
        for i, data in enumerate(dataloader, 1):
            input_ = data["data"].to(device)
            labels = data["label"].to(device, dtype=torch.float)
            batch_size = labels.size(0)
            labels = labels.reshape(batch_size, 1, 1, 1)
            output = dnet(input_)
            dnet_error = scorer(output, labels)
            dnet_error.backward()
            optimizer.step()
            perc_done = (100*i)//num_batches
            status_bar = '='*perc_done + ' '*(100-perc_done)
            sys.stdout.write(f"Epoch {n:05d} Batch {i:05d}: [{status_bar}]"
                             f"{perc_done:02d}% \r")
            sys.stdout.flush()

        print(f"\nEpoch {n:05d}: Discriminator MSE = {dnet_error}")


if __name__ == '__main__':
    main()
