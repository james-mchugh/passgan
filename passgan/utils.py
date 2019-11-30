"""Utilities for PassGan

"""

# ----------------------------------------------------------------------
# imports
# ----------------------------------------------------------------------

# System level imports
import string
import math
import functools

# Third party imports
import typing
import numpy as np

# ----------------------------------------------------------------------
# globals
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# functions
# ----------------------------------------------------------------------

def vectorize_string(input_str: str,
                     num_chars: typing.Optional[int] = 0) -> np.array:
    """Create matrix from string.

    Create a matrix from the string where each column is an embedding
    for a character.


    Parameters
    ----------
    input_str : str
        String to vectorize.
    num_chars : optional, int
        Number of columns in the output vector. If more than the number
        of chars in the string, pad with zero vectors. If less than,
        truncate the string. If zero, match the number of chars in the
        input string. (the default is 0)

    Returns
    -------
    np.array
        Matrix representation of string.

    """
    if num_chars == 0:
        num_chars = len(input_str)

    embedding_shape = _get_embedding_shape()
    string_matrix = np.zeros((embedding_shape[0], num_chars),
                             dtype=np.float32)
    for i, char in enumerate(input_str):
        string_matrix[:, i] = _embed_char(char)

    return string_matrix


def _embed_char(char: str) -> np.array:
    char_bin = [int(c) for c in f"{ord(char):07b}"]
    return np.array(char_bin).astype(int)


@functools.lru_cache(1)
def _get_embedding_shape() -> typing.Tuple[int, int]:
    max_ord = max([ord(char) for char in string.printable])
    embedding_size = math.ceil(math.log2(max_ord))
    return embedding_size, 1


def unvectorize_string(string_matrix: np.array) -> str:
    """Convert embedded string matrix back to string.

    Parameters
    ----------
    string_matrix : np.array
        Matrix of embedded vectors as columns to convert to string.

    Returns
    -------
    str
        String from the input matrix.

    """
    char_list = []
    for i in range(string_matrix.shape[1]):
        char_list.append(_unembed_char(string_matrix[:, i]))

    return "".join(char_list)


def _unembed_char(vector: np.array) -> str:
    ord_val = 0
    for i, val in enumerate(vector):
        ord_val += int(val * pow(2, i))

    return chr(ord_val)


def is_printable(input_str: str) -> bool:
    """Check if the string is printable.

    Parameters
    ----------
    input_str : str
        String to check if printable.

    Returns
    -------
    bool
        True if string contains printable characters.

    """
    return not set(input_str) - _get_printable()


@functools.lru_cache(1)
def _get_printable() -> typing.Set[str]:
    return set(string.printable)
