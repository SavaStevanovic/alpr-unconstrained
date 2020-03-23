import tensorflow as tf
from datetime import datetime
import os

def get_callbacks():
    callbacks = []
    callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath='./checkpoints/cp.ckpt', monitor='val_loss', verbose=0, save_best_only=True, save_freq='epoch'))
    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10))
    log_dir = os.path.join('logs', str(datetime.now().time()))
    callbacks.append(tf.keras.callbacks.TensorBoard(log_dir= log_dir, write_graph=True, update_freq='epoch', profile_batch=0))

    return callbacks

