import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras import Model
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Input
from tensorflow.keras.utils import plot_model

def WideAndDeepModel(Model):
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = Dense(units, activation='relu')
        self.hidden2 = Dense(units, activation='relu')
        self.main_output = Dense(1)
        self.aux_output = Dense(1)


    def call(self, inputs):
        input_A, Input_B = inputs
        hidden1 = self.hidden1(Input_B)
        hidden2 = self.hidden2(hidden1)
        concat = concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(main_output)

        return main_output, aux_output

model = WideAndDeepModel()