"""
This module contains functions specific to training, loading, and
saving neural network models.
"""
import datetime
import pickle
from typing import Tuple
import tensorflow as tf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import window_generator


MAX_EPOCHS = 100
MODEL_LOCATION = "models/"


def compile_and_fit(model: tf.python.keras.engine.sequential.Sequential,
                    window: window_generator.WindowGenerator, patience: int = 2,
                    max_epochs: int = MAX_EPOCHS,
                    log: bool = False) -> tf.python.keras.callbacks.History:
    """
    Returns history object and acts as a training procedure for models.
    Logging requires tensorboard.

    - patience > 0
    - max_epochs > 0
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    if log:
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(
            log_dir=log_dir, histogram_freq=1)

        history = model.fit(window.train, epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping, tensorboard_callback])
    else:
        history = model.fit(window.train, epochs=max_epochs,
                            validation_data=window.val,
                            callbacks=[early_stopping])
    return history


def save_model(model: tf.python.keras.engine.sequential.Sequential,
               history: dict,
               filename: str,
               location: str = MODEL_LOCATION) -> None:
    """
    Saves model to a specified location with a specified name.
    """
    print(f"SAVING model to {location} as {filename}")
    model.save(location + filename)
    with open(location + filename + "/history.p", 'wb') as file_pi:
        pickle.dump(history, file_pi)


def load_model(filename: str, location: str = MODEL_LOCATION
               ) -> Tuple[tf.keras.Model, tf.python.keras.callbacks.History]:
    """
    Loads model from a specified location with a specified name.
    """
    print(f"LOADING model {filename} from {location}")
    model = tf.keras.models.load_model(location + filename)
    with open(location + filename + "/history.p", 'rb') as file_pi:
        history = pickle.load(file_pi)
    return model, history


def evaluate_model(model: tf.python.keras.engine.sequential.Sequential,
                   window: window_generator.WindowGenerator,
                   history: dict = None,
                   verbose: int = 0) -> None:
    """
    Displays information about the model and its training history.

    Verbose = 0 for only basic information about the final model.
    Verbose = 1 for information about the final model and training history.
    Verbose = 2 for information about the final model, training history, and shape.

    - verbose in [0, 1, 2]
    """
    final_model_evaluation(model, window)
    if verbose == 1:
        model_history_evaluation(history)
    elif verbose == 2:
        model_history_evaluation(history)
        print(model.summary())


def model_history_evaluation(history: dict) -> None:
    """
    Plots history of model loss, val loss, mean absolute error, and va mean absolute error
    from training.

    - history is not None
    """
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=('Loss During Training',
                                        'Mean Absolute Error During Training'))

    fig.add_trace(go.Scatter(y=history['loss'],
                             mode='lines',
                             name='Loss'), row=1, col=1)
    fig.add_trace(go.Scatter(y=history['val_loss'],
                             mode='lines',
                             name='Valuation Loss'), row=1, col=1)

    fig.update_yaxes(title_text='Loss', row=1, col=1)
    fig.update_xaxes(title_text='Epoch', row=1, col=1)

    fig.add_trace(go.Scatter(y=history['mean_absolute_error'],
                             mode='lines',
                             name='Mean Absolute Error'), row=2, col=1)
    fig.add_trace(go.Scatter(y=history['val_mean_absolute_error'],
                             mode='lines',
                             name='Valuation Mean Absolute Error'), row=2, col=1)

    fig.update_yaxes(title_text='Mean Absolute Error', row=2, col=1)
    fig.update_xaxes(title_text='Epoch', row=2, col=1)

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
            l=0,  # left margin
            r=0,  # right margin
        )
    )

    fig.show()


def final_model_evaluation(model: tf.python.keras.engine.sequential.Sequential,
                           window: window_generator.WindowGenerator) -> None:
    """
    Plots final model loss and mean absolute error for valuation and testing datasets.
    """
    val = model.evaluate(window.val)
    test = model.evaluate(window.test, verbose=0)

    fig = go.Figure([go.Bar(name='Loss', x=['val', 'test'], y=[val[0], test[0]]),
                     go.Bar(name='Mean Absolute Error',
                            x=['val', 'test'], y=[val[1], test[1]])])

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
        'extra-imports': ['datetime', 'tensorflow',
                          'python_ta.contracts',
                          'window_generator',
                          'plotly.graph_objects',
                          'pickle', 'typing', 'Tuple',
                          'plotly.subplots'],
        'allowed-io': ['save_model', 'load_model', 'evaluate_model'],
        'max-line-length': 100,
        'disable': ['R1705', 'C0200']
    })

    import python_ta.contracts
    python_ta.contracts.DEBUG_CONTRACTS = False
    python_ta.contracts.check_all_contracts()

    import doctest
    doctest.testmod()
