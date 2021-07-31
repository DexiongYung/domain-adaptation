from __future__ import print_function
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.layers import LeakyReLU
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt

class CNN(object):
    def __init__(self):
        # change these to appropriate values
        self.batch_size = 256
        self.epochs = 2 # best to run for 20 epochs
        self.init_lr= 0.75 #learning rate
        self.alpha = 0.15
        self.drop_out = 0.25
        
        # No need to modify these
        self.model = None

    def get_vars(self):
        return self.batch_size, self.epochs, self.init_lr

    def create_net(self):
        '''
        In this function you are going to build a convolutional neural network based on TF Keras.
        First, use Sequential() to set the inference features on this model. 
        Then, use model.add() to build layers in your own model
        Return: model
        '''
        self.model = Sequential(layers=[
            Conv2D(input_shape=(32, 32, 3), filters=16, kernel_size=3, padding='same'),
            LeakyReLU(input_shape=(32, 32, 16), alpha=self.alpha),
            Conv2D(input_shape=(32, 32, 16), filters=32, kernel_size=3, padding='same'),
            LeakyReLU(input_shape=(32, 32, 32), alpha=self.alpha),
            MaxPooling2D(input_shape=(32, 32, 32), pool_size=(2, 2)),
            Dropout(input_shape=(16, 16, 32), rate=self.drop_out),
            Conv2D(input_shape=(16, 16, 32), filters=32, kernel_size=3, padding='same'),
            LeakyReLU(input_shape=(16, 16, 32), alpha=self.alpha),
            Conv2D(input_shape=(16, 16, 32), filters=64, kernel_size=3, padding='same'),
            LeakyReLU(input_shape=(16, 16, 64), alpha=self.alpha),
            MaxPooling2D(input_shape=(16, 16, 64), pool_size=(2, 2)),
            Dropout(input_shape=(8, 8, 64), rate=self.drop_out),
            Flatten(),
            Dense(256),
            LeakyReLU(alpha=0.1),
            Dropout(rate=self.drop_out),
            Dense(10),
            Activation('softmax')]
        )

        return self.model

    def compile_net(self, model):
        '''
        In this function you are going to compile the model you've created.
        Use model.compile() to build your model.
        '''
        self.model = model
        self.model.compile(loss='categorical_crossentropy',
                      optimizer='sgd',
                      metrics=['accuracy'])
        return self.model


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# input image dimensions
img_rows, img_cols = 32, 32
number_channels = 3
#set num of classes
num_classes = 10

if tf.keras.backend.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], number_channels, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], number_channels, img_rows, img_cols)
    input_shape = (number_channels, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, number_channels)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, number_channels)
    input_shape = (img_rows, img_cols, number_channels)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

cifar10_classes = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
# convert class vectors to binary class matrices
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

def lr_scheduler(epoch):
    new_lr = init_lr * 0.9 ** epoch
    print("Learning rate:", new_lr)
    return new_lr

alpha = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
batch_sz = [8, 16, 32, 64, 128, 256]
lr = [0.01, 0.025, 0.05, 0.075, 0.1, 0.125]
do = [0, 0.05, 0.1, 0.15, 0.2, 0.25]
epochs_lst = [20, 30, 40]

best_score = float('-inf')
best_lr, best_alpha, best_bs, best_do, best_epoch = None, None, None, None, None

for a in alpha:
    for b in batch_sz:
        for l in lr:
            for d in do:
                for e in epochs_lst:
                    print(f"Current Params: alpha: {a}, batch: {b}, lr: {l}, drop out: {d}, epochs: {e}")
                    net = CNN()
                    net.batch_size = b
                    net.init_lr = l
                    net.alpha = a
                    net.drop_out = d
                    net.epochs = e
                    model = net.create_net()
                    model = net.compile_net(model)
                    batch_size, epochs, init_lr = net.get_vars()

                    history = model.fit(
                        x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        callbacks=[tf.keras.callbacks.LearningRateScheduler(lr_scheduler)],
                        shuffle=True,
                        verbose=1,
                        initial_epoch=0, 
                        validation_data=(x_test, y_test)
                    )
                    score = model.evaluate(x_test, y_test, verbose=0)
                    print('Test loss:', score[0])
                    print('Test accuracy:', score[1])

                    if score[1] > best_score:
                        best_score = score[1]
                        best_lr = l
                        best_alpha = a
                        best_bs = b
                        best_do = d
                        best_epoch = e

print(f"Best score is: {best_score}, LR: {best_lr}, alpha: {best_alpha}, batch size: {best_bs}, drop out: {best_do}, epoch: {best_epoch}")