import pickle
import numpy
import tensorflow as tf
# TODO: import Keras layers you need here
from keras.models import Sequential
from keras.layers import Flatten, Dense

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('network', '', "inception|resnet|vgg")
flags.DEFINE_string('dataset', '', "traffic|cifar10")

def load_bottleneck_data(network_name, dataset_name):
    """
    Utility function to load bottleneck features.

    Arguments:
        network_name = String
        dataset_name - String
    """
    print("Using network:", network_name)
    print("Using dataset:", dataset_name)

    with open(network_name + "_" + dataset_name + "_100_bottleneck_features_train.p", 'rb') as f:
        train_data = pickle.load(f)
    with open(network_name + "_" + dataset_name + "_bottleneck_features_validation.p", 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.network, FLAGS.dataset)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    nb_classes = len(numpy.unique(y_train))

    # TODO: define your model and hyperparams here
    model = Sequential()
    # make sure to adjust the number of classes based on
    # the dataset
    # 10 for cifar10
    # 43 for traffic
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(nb_classes, activation='softmax'))

    # TODO: train your model here
    model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
    history = model.fit(X_train, y_train, batch_size=256, nb_epoch=50, validation_data=(X_val, y_val), shuffle=True)

# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
