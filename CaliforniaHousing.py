import numpy as np 
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf
import sklearn
# from sklearn.datasets.california_housing import fetch_california_housing
[dataset for dataset in sklearn.datasets.__dict__.keys() if str(dataset).startswith("fetch_") or str(dataset).startswith("load_")]

from sklearn.datasets.california_housing import fetch_california_housing

print("Hello")
housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="x")
y = tf.constant(housing.target.reshape(-1,1), dtype=tf.float32, name='y')

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

with tf.Session() as sess:
    theta_value = theta.eval()
    print(theta_value)