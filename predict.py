"""
This module includes functions used to utilized trained neural network
models in order to obtain an output.
"""
from typing import List
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import plotly.graph_objects as go
import statsmodels.api as sm

import window_generator


def predict_plot(model: tf.python.keras.engine.sequential.Sequential,
                 window: window_generator.WindowGenerator, label: str,
                 matplot: bool = False) -> None:
    """
    Plots a graph of a feature predictions on top of the real feature values.
    Primarily uses plotly but has matplotlib as a backup.

    - label in window.label_columns or window.label_columns == None
    """
    plot_col_index = window.column_indices[label]
    prediction = model.predict(window.ground)
    ground = prediction[:, -1, plot_col_index]

    val = np.empty_like(window.dfs.ground_df[label])
    val.fill(np.nan)
    val[window.ws.total_window_size - 1:] = ground

    plot(val, window, label, matplot=matplot)


def plot(val: List[float], window: window_generator.WindowGenerator,
         label: str, matplot: bool = False) -> None:
    """
    Produces layred line plots using the given information. Produces line of best fit
    when using the plotly graph.

    - len(val) == len(ground)
    """
    if matplot:
        plt.clf()
        plt.plot(window.dfs.ground_df[label])
        plt.plot(val, alpha=0.5)
        plt.show()
    else:
        # Line of best fit calculations
        x = list(range(window.ws.input_width, len(val[:])))
        df = pd.DataFrame({'X': x, 'Y': val[window.ws.input_width:]})
        sm_results = sm.OLS(
            df['Y'], sm.add_constant(df['X'])).fit()
        df['bestfit'] = sm_results.fittedvalues
        slope = sm_results.params['X']

        # Draw graph
        fig = go.Figure()
        fig.update_layout(
            xaxis_title='Time [Hours]',
            yaxis_title='Normalized ' + label,
            font=dict(
                family="Courier New, monospace",
                size=12),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ), margin=go.layout.Margin(
                l=0,  # left margin
                r=0,  # right margin
                b=0,  # bottom margin
                t=0  # top margin
            )
        )

        fig.add_trace(go.Scatter(y=window.dfs.ground_df[label],
                                 mode='lines',
                                 name='True ' + label))
        fig.add_trace(go.Scatter(y=val,
                                 mode='lines',
                                 name='Model Predictions'))
        fig.add_trace(go.Scatter(name='MP - Line of Best Fit : ' + str(slope),
                                 x=x, y=df['bestfit'], mode='lines'))
        fig.show()


if __name__ == '__main__':
    import python_ta
    python_ta.check_all(config={
        # the names (strs) of imported modules
        'extra-imports': ['pandas', 'window_generator', 'matplotlib.pyplot',
                          'numpy', 'python_ta.contracts', 'statsmodels.api',
                          'tensorflow', 'plotly.graph_objects'],
        'allowed-io': [],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
