import tensorflow as tf

class Callback(object):
    def __init__(self):
        self.validation_data = None
        self.model = None
    
    def on_epoch_begin(self, epoch, logs=None):
        ""
    
    def on_epoch_end(self, epoch, logs=None):
        ""

    def on_(train|test|predict)_begin(self, logs=None):
        ""
    
    def on_(train|test|predict)_end(self, logs=None):
        ""

    def on_(train|test|predict)_batch_begin(self, logs=None):
        ""
    
    def on_(train|test|predict)_batch_end(self, logs=None):
        ""