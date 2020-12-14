"""
The module includes functions that generates the
model structures as well as their accompanying window
generator.
"""
from typing import Tuple, List
import tensorflow as tf
from window_generator import WindowGenerator
from data_classes import DataFrameSet, WindowSet


def single_out(dfs: DataFrameSet, units: int = 32, labels: List[str] = None,
               input_steps: int = 24,
               model_type: str = 'lstm') -> Tuple[tf.keras.models.Sequential, WindowGenerator]:
    """
    Returns model and window generator designed for outputting a single step. Can handle multiple
    labels.

    - units > 0
    - input_steps > 0
    - model_type in ['lstm', 'gru']
    """
    num_features = dfs.ground_df.shape[1]
    if labels is not None:
        output_units = len(labels)
    else:
        output_units = num_features
    if model_type == 'lstm':
        single_model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=True),
            tf.keras.layers.Dense(units=output_units)
        ])
    else:
        single_model = tf.keras.models.Sequential([
            tf.keras.layers.GRU(units, return_sequences=True),
            tf.keras.layers.Dense(units=output_units)
        ])

    ws = WindowSet(input_width=input_steps, label_width=input_steps, shift=1)

    window = WindowGenerator(dfs=dfs, ws=ws, label_columns=labels)

    return single_model, window


def multi_out(dfs: DataFrameSet, units: int = 32, input_steps: int = 24, out_steps: int = 24,
              model_type: str = 'lstm') -> Tuple[tf.keras.models.Sequential, WindowGenerator]:
    """
    Returns model and window generator designed for outputting a multiple steps. Outputs all labels.

    - units > 0
    - out_steps > 0
    - input_steps > 0
    - model_type in ['lstm', 'gru']
    """
    num_features = dfs.ground_df.shape[1]
    if model_type == 'lstm':
        multi_model = tf.keras.Sequential([
            tf.keras.layers.LSTM(units, return_sequences=False),
            tf.keras.layers.Dense(out_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([out_steps, num_features])
        ])
    else:
        multi_model = tf.keras.Sequential([
            tf.keras.layers.GRU(units, return_sequences=False),
            tf.keras.layers.Dense(out_steps * num_features,
                                  kernel_initializer=tf.initializers.zeros),
            tf.keras.layers.Reshape([out_steps, num_features])
        ])

    ws = WindowSet(input_width=input_steps,
                   label_width=out_steps, shift=out_steps)

    window = WindowGenerator(dfs=dfs, ws=ws)

    return multi_model, window


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['WindowSet', 'tensorflow', 'python_ta.contracts',
                          'Tuple', 'List', 'data_classes', 'WindowGenerator',
                          'window_generator'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
