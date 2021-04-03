import tensorflow as tf

print(tf.__version__)


#Version 1.15

#Can run this in tensorflow 2.0 using the following code
# import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
# Howerver this is going to be depreceated. 

#creating the variables
x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y +2

#running the session where the computation power happens
# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()

# other way to run the session 
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
#     print(result)

#rather than Initializing the variables one by one. 
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    result = f.eval()
    print(result)

#Interactive session 
# sess = tf.InteractiveSession()

x1 = tf.Variable(1)
x1.graph is tf.get_default_graph()

#Creating a temp graph to differentitate between the existing graph and new one
graph = tf.Graph()
with graph.as_default():
    x2 = tf.Variable(2)

x2.graph is graph

x2.graph is tf.get_default_graph()

tf.reset_default_graph()
