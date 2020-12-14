"""
Module hosting a generator that creates the windows being fed into the
neural network. It also hosts information about the window and the
data being fed into the neural network.
"""
from typing import Tuple

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from data_classes import DataFrameSet, WindowSet, WindowInputs, WindowLabels


class WindowGenerator():
    """
    Generator for window inputs into a model for training, validation, testing
    and predictions.
    """
    dfs: DataFrameSet
    label_columns: list
    label_columns_indices: dict
    column_indices: dict
    ws: WindowSet
    inputs: WindowInputs
    labels: WindowLabels

    def __init__(self, dfs: DataFrameSet,
                 ws: WindowSet,
                 label_columns: list = None) -> None:
        # Store the raw data.
        self.dfs = dfs

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(dfs.train_df.columns)}

        # Work out the window parameters.
        self.ws = ws

        self.inputs = WindowInputs(self.ws)
        self.labels = WindowLabels(self.ws)

    def __repr__(self) -> None:
        return '\n'.join([
            f'Total window size: {self.ws.total_window_size}',
            f'Input indices: {self.inputs.input_indices}',
            f'Label indices: {self.labels.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features: tf.Tensor) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        """
        Returns a tuple of Datasets with one being inputs for a window and the other
        being labels for a window.
        """
        inputs = features[:, self.inputs.input_slice, :]
        labels = features[:, self.labels.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]]
                 for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.ws.input_width, None])
        labels.set_shape([None, self.ws.label_width, None])

        return inputs, labels

    def make_dataset(self, data: pd.core.frame.DataFrame,
                     shuffle: bool = True) -> tf.python.data.ops.dataset_ops.MapDataset:
        """
        Returns takes a times series DataFrame and converts it into a tf Dataset
        of input window and label window pairs.
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.ws.total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=512,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self) -> tf.python.data.ops.dataset_ops.MapDataset:
        """Access property for training dataset"""
        return self.make_dataset(self.dfs.train_df.values)

    @property
    def val(self) -> tf.python.data.ops.dataset_ops.MapDataset:
        """Access property for validation dataset"""
        return self.make_dataset(self.dfs.val_df.values)

    @property
    def test(self) -> tf.python.data.ops.dataset_ops.MapDataset:
        """Access property for testing dataset"""
        return self.make_dataset(self.dfs.test_df.values)

    @property
    def ground(self) -> tf.python.data.ops.dataset_ops.MapDataset:
        """Access property for ground dataset"""
        return self.make_dataset(self.dfs.ground_df.values, shuffle=False)

    @property
    def example(self) -> tuple:
        """Get an example batch of `inputs, labels` for plotting."""
        result = next(iter(self.test))
        return result

    def plot(self, model: tf.python.keras.engine.sequential.Sequential = None,
             plot_col: str = 't2m', max_subplots: int = 3) -> None:
        """
        Generate graph(s) that shows the visualization of a split window, with
        predicted datapoints overlayed when avaliable.

        - plot_col in self.column_indices
        """
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))

        fig = make_subplots(rows=max_n, cols=1, x_title='Time [hours]',
                            y_title='Normalized ' + plot_col,)

        for n in range(1, max_n + 1):

            fig.add_trace(go.Scatter(x=self.inputs.input_indices,
                                     y=inputs[n, :,
                                              plot_col_index], mode='lines+markers',
                                     name=str('Inputs ' + str(n))), row=n, col=1)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(
                    plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            fig.add_trace(go.Scatter(x=self.labels.label_indices,
                                     y=labels[n, :, label_col_index],
                                     mode='markers',
                                     name=str('Labels ' + str(n))),
                          row=n, col=1)
            if model is not None:
                predictions = model(inputs)
                fig.add_trace(go.Scatter(x=self.labels.label_indices,
                                         y=predictions[n, :, label_col_index],
                                         mode='markers',
                                         name=str('Predictions ' + str(n))),
                              row=n, col=1)
        fig.update_layout(
            font=dict(
                family="Courier New, monospace",
                size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ), margin=go.layout.Margin(
                r=0,  # right margin
                t=0  # top margin
            )
        )

        fig.show()


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['DataFrameSet', 'tensorflow', 'matplotlib.pyplot',
                          'numpy', 'python_ta.contracts', 'Tuple',
                          'WindowSet', 'data_classes', 'pandas',
                          'plotly.subplots', 'plotly.graph_objects', ],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
