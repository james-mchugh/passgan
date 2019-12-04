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
import logging

# Third party imports
import torch
import sklearn.metrics
from torch.utils.data import DataLoader

# Local application imports
from passgan.datasets.linkedin import PasswordDataset
from passgan.models.discriminator import Discriminator
from passgan import utils


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
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

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)

    seed = args.seed if args.seed else random.randint(1, 200000)
    logger.info(f"Setting manual seed to {seed}.")
    torch.manual_seed(seed)

    logger.info("Reading and Processing Data.")
    dataloader = DataLoader(PasswordDataset(), batch_size=args.batch_size,
                            num_workers=args.workers, shuffle=True,
                            pin_memory=True)

    device = torch.device("cuda:0") if args.ngpu > 0 else "cpu"

    logger.info("Initializing discriminator.")
    dnet = Discriminator(args.ngpu).to(device)
    dnet.apply(utils.init_weights)
    optimizer = torch.optim.Adam(dnet.parameters(), lr=args.lr)

    loss_function = torch.nn.BCELoss()

    num_batches = len(dataloader)

    logger.info(f"Starting training for {args.epochs} epochs with a batch "
                f"size of {args.batch_size}")
    for n in range(args.epochs):
        for i, data in enumerate(dataloader, 1):
            dnet.zero_grad()
            input_ = data["data"].to(device)
            labels = data["label"].to(device, dtype=torch.float)
            output = dnet(input_)
            dnet_error = loss_function(output, labels)
            dnet_error.backward()
            optimizer.step()
            perc_done = (100*i)//num_batches
            status_bar = '='*perc_done + ' '*(100-perc_done)
            sys.stdout.write(f"Epoch {n:04d} - Batch {i:05d}: [{status_bar}]"
                             f"{perc_done:02d}% \r")
            sys.stdout.flush()

        accuracy = sklearn.metrics.accuracy_score(labels, output)
        logger.info(f"\nAccuracy: {accuracy}")
        logger.info(f"\nEpoch {n:05d}: Discriminator {loss_function.__class__.__name__} = {dnet_error}")


if __name__ == '__main__':
    main()
