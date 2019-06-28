import os

import numpy as np
import pytest
import six
from keras import backend as K
from keras import utils
from keras_applications.imagenet_utils import decode_predictions
from keras_preprocessing.image import img_to_array, load_img

import keras_efficientnets as KE


def reset_backend(func):
    @six.wraps(func)
    def wrapped(*args, **kwargs):
        K.clear_session()
        K.reset_uids()
        return func(*args, **kwargs)

    return wrapped


def get_preds(model):
    size = model.input_shape[1]

    filename = os.path.join(os.path.dirname(__file__),
                            'data', '565727409_61693c5e14.jpg')

    batch = KE.preprocess_input(img_to_array(load_img(
                                filename, target_size=(size, size))))

    batch = np.expand_dims(batch, 0)

    pred = decode_predictions(model.predict(batch),
                              backend=K, utils=utils)

    return pred


@reset_backend
def test_efficientnet_b0():
    model = KE.EfficientNetB0(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


@reset_backend
def test_efficientnet_b1():
    model = KE.EfficientNetB1(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


@reset_backend
def test_efficientnet_b2():
    model = KE.EfficientNetB2(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


@reset_backend
def test_efficientnet_b3():
    model = KE.EfficientNetB3(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


@reset_backend
def test_efficientnet_b4():
    model = KE.EfficientNetB4(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


@reset_backend
def test_efficientnet_b5():
    model = KE.EfficientNetB5(weights='imagenet')
    assert model is not None

    pred = get_preds(model)
    assert pred[0][0][1] == 'tiger_cat'


if __name__ == '__main__':
    pytest.main(__file__)
