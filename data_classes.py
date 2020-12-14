"""
This module contains the custom data type DataFrameSet and
WindowSet.
"""
from dataclasses import dataclass
import pandas as pd
import numpy as np


@dataclass
class DataFrameSet:
    """A custom data type which represents the four dataframes used
    in window generation and training. train_df, val_df, test_df and
    ground_df.
    """
    train_df: pd.core.frame.DataFrame
    val_df: pd.core.frame.DataFrame
    test_df: pd.core.frame.DataFrame
    ground_df: pd.core.frame.DataFrame


class WindowSet:
    """A data type which represents the four highest level attributes
    of a window. input_width, label_width, shift and total_window_size.

    - input_width > 0
    - label_width > 0
    - shift > 0
    """
    input_width: int
    label_width: int
    shift: int
    total_window_size: int

    def __init__(self, input_width: int, label_width: int,
                 shift: int) -> None:
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        self.total_window_size = input_width + shift


class WindowInputs:
    """A data type which represents window input attributes.
    """
    input_slice: slice
    input_indices: np.ndarray

    def __init__(self, ws: WindowSet) -> None:
        self.input_slice = slice(0, ws.input_width)
        self.input_indices = np.arange(ws.total_window_size)[
            self.input_slice]


class WindowLabels:
    """
    A data type which represents window label attributes.
    """
    label_start: int
    labels_slice: slice
    label_indices: np.ndarray

    def __init__(self, ws: WindowSet) -> None:
        self.label_start = ws.total_window_size - ws.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(ws.total_window_size)[
            self.labels_slice]


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['pandas', 'dataclass', 'dataclasses',
                          'numpy', 'python_ta.contracts'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
