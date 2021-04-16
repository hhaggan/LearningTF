import tensorflow as tf 

tf.executing_eagerly()

a = tf.constant(5)
b = a * 3

print(b)

defaults = [tf.int32] * 55
dataset = tf.data.CsvDataset(['covtype.csv.train'], defaults)