import tensorflow as tf


#residual Hands on 1

# class MyResidual(Model):
#     def __init__(**kwargs):
#         # super().__init__(**kwargs)
#         self.hidden1 = Dense(30, activation='relu')
#         self.block1 = CNNResidual(2, 32)
#         self.block2 = DNNResidual(2, 64)
#         self.out = Dense(1)

#     def call(self, inputs):
#         x = self.hidden1(inputs)
#         x = self.block1(x)

#         for _ in range(1,4):
#             x = self.block2(x)
        
#         return self.out(x)

# model = MyResidual()

#Residual Hands on 2

class IdentityBlock(tf.keras.Model):
    def __init__(self, filters, kernel_size):
        super(IdentityBlock, self).__init__(name='')
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn1 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = tf.keras.layers.BatchNormalization()

        self.act = tf.keras.activations('relu')
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor):
        x = self.conv1(input_tensor)
        x = self.bn1(x)
        x = self.act(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.act(x)

        x = self.add([x, input_tensor])
        x = self.act(x)

        return x

class ResNet(tf.keras.Model):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        self.conv = tf.keras.layers.Conv2D(64, 7, padding='same')
        self.bn = tf.keras.layers.BatchNormalization()
        self.act = tf.keras.layers.Activation('relu')
        self.max_pool = tf.keras.layers.MaxPool2D((3,3))
        self.id1a = IdentityBlock(64,3)
        self.id1b = IdentityBlock(64,3)
        self.global_pool = tf.keras.layers.GlobalAveragePooling2D()
        self.classifier = tf.keras.layers.Dense(num_classes, activation='relu')

    def call(self, inputs):
        x = self.conv(inputs)
        x = self.bn(x)
        x = self.act(x)

        x = self.max_pool(x)
        x = self.id1a(x)
        x = self.id1b(x)
        x = self.global_pool(x)

        return self.classifier(x)

def preprocess(features):
    return tf.cast(features["image"], tf.float32)/255., features['label']

resnet = ResNet(10)
resnet.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

dataset = tfds.load('mnist', split=tfds.Split.TRAIN)

dataaset = dataaset.map(preprocess).batch(32)
resent.fit(dataset, epochs=1)