"""
    Tensorflow's cnn model
"""

from algorithm.cnn import Convolution2D, FullyConnected, MaxPool, ReLU, Softmax, CNNModel
from tensorflow import train


def train_model(model_name):
    """
        train a model and save it to folder '/trained_model/model_name'
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
    model = CNNModel(image_size=[28, 28], char_number=10, channel=1)

    model.addLayer(Convolution2D(size=[5, 5], features=32))
    model.addLayer(ReLU())
    model.addLayer(MaxPool(size=[2, 2]))

    model.addLayer(Convolution2D(size=[5, 5], features=64))
    model.addLayer(ReLU())
    model.addLayer(MaxPool(size=[2, 2]))

    model.addLayer(FullyConnected(features=1024))
    model.addLayer(ReLU())

    # model.addLayer(FullyConnected(features=512))
    # model.addLayer(ReLU())

    model.addLayer(FullyConnected(features=10))
    model.addOutputLayer(Softmax())

    model.train(
        dataset=mnist,
        eval_every=100,
        epochs=20000,
        evaluation_size=100,
        batch_size=50,
        optimizer=train.AdamOptimizer(0.0001))

    model_path = "trained_model/" + model_name + "/" + model_name
    model.save(model_path)


train_model("cnn-tensorflow")
