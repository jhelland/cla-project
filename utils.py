"""
General utility functions
"""
#########################
# LIBRARIES
#########################

# data wrangling
import six.moves.cPickle as pickle
import gzip
import os

# computations & neural network methods/structures
import numpy as np
import theano
import theano.tensor as T


#########################
# FUNCTIONS
#########################

def load_data(dataset):
    """
    Loads the MNIST dataset or downloads if not found
    :type dataset: string
    :param dataset: the path to the dataset (here MNIST)
    """

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        # Check if dataset is in the data directory.
        new_path = os.path.join(
            os.path.split(__file__)[0],
            "..",
            "data",
            dataset
        )
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':
            dataset = new_path

    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

    print('...Loading data')

    # Load the dataset
    with gzip.open(dataset, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
    # train_set, valid_set, test_set format: tuple(input, target)
    # input is a numpy.ndarray of 2 dimensions (a matrix)
    # where each row corresponds to an example. target is a
    # numpy.ndarray of 1 dimension (vector) that has the same length as
    # the number of rows in the input. It should give the target
    # to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval


def save_model_parameters_theano(model, outfile):
    """
    Save model parameters to .npz file. Currently only set up to work with MLP class model.
    :param model: object containing model parameters
    :param outfile: string of file output path
    """
    params = [param.get_value() for param in model.params]
    np.savez(outfile,
             *params)
    print("...Saved model parameters to %s." % outfile)


def load_model_parameters_theano(path, modelClass):
    """
    Load the model parameters from a .npz file.
    :param path: string of path to .npz file
    :param modelClass: handle to desired neural network class
    :return: built model
    """

    npzfile = np.load(path)

    W = []
    b = []
    count = 0
    for key in sorted(npzfile):
        if count % 2 is 0:
            W.append(npzfile[key])
        else:
            b.append(npzfile[key])
        count += 1

    layer_sizes = [param.shape[0] for param in W]
    layer_sizes.append(b[-1].shape[0])

    model = modelClass(
        rng=np.random.RandomState(1234),
        input=T.matrix('x'),
        layer_sizes=layer_sizes,
        W=W,
        b=b
    )

    return model


def load_conv_net(path, pool_sizes, image_size, batch_size):

    from nnet_classes import CNN

    npzfile = np.load(path)

    W = []
    b = []
    _layer_sizes = []
    _filter_sizes = []
    _pool_sizes = pool_sizes
    count = 0
    for key in sorted(npzfile):
        if count % 2 == 0:
            W.append(npzfile[key])
            if len(W[-1].shape) > 2:
                _layer_sizes.append(W[-1].shape[0])
                _filter_sizes.append(W[-1].shape[-2:])
            else:
                _layer_sizes.append(W[-1].shape[-1])
            print('W', W[-1].shape)
        else:
            b.append(npzfile[key])
            print('b', b[-1].shape)
        count += 1

    rng = np.random.RandomState(None)
    layer0_input = T.matrix('x').reshape((batch_size, 1, *image_size))
    return CNN(rng=rng,
               input=layer0_input,
               layer_sizes=_layer_sizes,
               image_size=(batch_size, 1, *image_size),
               filter_sizes=_filter_sizes,
               pool_sizes=_pool_sizes,
               W=W,
               b=b,
               activations=None)



# vectorize a matrix
def vec(matrices, axis=1):
    return np.concatenate([a.T.flatten() if axis else a.flatten()
                           for a in matrices])


# recover original model parameter structure
def format_params(params_vec, dims):
    from functools import reduce
    from operator import mul

    params = []
    j = 0
    for dim in dims:
        n = reduce(mul, dim)
        param = params_vec[j:(j+n)]
        if len(dim) > 1:
            param = param.reshape(dim[1], dim[0])
        params.append(param.T)
        j += n
    return params
