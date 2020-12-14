"""
The main file for the CSC110 term project.
See report for usage instructions. Requires
a 64bit version of python 3.8. CUDA based
GPU acceleration optional.
"""
import tensorflow as tf

import preprocess
import predict
import model
import train


def enable_mem_growth() -> None:
    """
    Enables memory growth for more stable GPU usage.
    Experimental feature for Tensorflow. Do not run if
    you are not using a GPU.
    """
    gpu_devices = tf.config.experimental.list_physical_devices('GPU')
    for device in gpu_devices:
        tf.config.experimental.set_memory_growth(device, True)


enable_mem_growth()

# Set to true to save processed data.
dfs = preprocess.ingest_new_data(
    cache_data=False)

# Used to ingest pre-processed data. Comment out ingest new data if using this.
#dfs = preprocess.ingest_cached_data(filename='')

# ==============================
# Model and window configuration
#
# Only run one line in this section!
# ===============================

model, window = model.single_out(dfs, input_steps=25, model_type='lstm')

# model, window = model.multi_out(
#    dfs, input_steps=25, out_steps=25, model_type='lstm')

# Load model history
# model, history = train.load_model('model')

# ==============================
# Training
# ===============================

history = train.compile_and_fit(model, window).history

# train.save_model(model, history, 'model')

# ==============================
# Predictions and Evaluations
# ===============================

train.evaluate_model(model, window, history, verbose=2)

window.plot(model, plot_col='t2m')

predict.predict_plot(model, window, label='t2m')
