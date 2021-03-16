"""
Collection of small helpers.

moduleauthor:: Alvis HT Tang <alvis@hilbert.space>
"""

from typing import Dict


def dict_to_str(dict: Dict) -> str:
    """
    Stringify a dictionary.

    Parameters
    ----------
    dict
        dictionary to be converted as a string

    Returns
    -------
        the dictionary represented as a string
    """
    return str.join(
        ",",
        [
            f"{key}={value if not callable(value) else (value.__name__ if hasattr(value, '__name__') else type(value).__name__) }"
            for key, value in dict.items()
        ],
    )
