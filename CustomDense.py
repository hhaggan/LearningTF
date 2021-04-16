from tensorflow.keras.layers import Layer

class SimpleDense(Layer):

    def __init__(self, units=32, activation="None"):
        '''Initializes the instance attributes'''
        super(SimpleDense, self).__init__()
        self.units = units
        self.activation = tf.keras.activation.get(activation)

    def build(self, input_shape):
        '''Create the state of the layer (weights)'''
        # initialize the weights
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(name="kernel",
            initial_value=w_init(shape=(input_shape[-1], self.units),
                                 dtype='float32'),
            trainable=True)

        # initialize the biases
        b_init = tf.zeros_initializer()
        self.b = tf.Variable(name="bias",
            initial_value=b_init(shape=(self.units,), dtype='float32'),
            trainable=True)

    def call(self, inputs):
        '''Defines the computation from inputs to outputs'''
        return self.activation(tf.matmul(inputs, self.w) + self.b)

# # declare an instance of the class
# my_dense = SimpleDense(units=1)

# # define an input and feed into the layer
# x = tf.ones((1, 1))
# y = my_dense(x)

# # parameters of the base Layer class like `variables` can be used
# print(my_dense.variables)

# #different way

# # define the dataset
# xs = np.array([-1.0,  0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
# ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)


# # use the Sequential API to build a model with our custom layer
# my_layer = SimpleDense(units=1)
# model = tf.keras.Sequential([my_layer])

# # configure and train the model
# model.compile(optimizer='sgd', loss='mean_squared_error')
# model.fit(xs, ys, epochs=500,verbose=0)

# # perform inference
# print(model.predict([10.0]))

# see the updated state of the variables
print(my_layer.variables)


#try MNIST Data

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    SimpleDense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)
model.evaluate(x_test, y_test)