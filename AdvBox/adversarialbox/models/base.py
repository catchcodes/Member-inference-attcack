"""
The base model of the model.
"""
from __future__ import division
from builtins import object
from past.utils import old_div
from abc import ABCMeta
from abc import abstractmethod

import numpy as np
from future.utils import with_metaclass

# 抽象类
class Model(with_metaclass(ABCMeta, object)):
    """
    Base class of model to provide attack.

    Args:
        bounds(tuple): The lower and upper bound for the image pixel.
        channel_axis(int): The index of the axis that represents the color
                channel.
        preprocess(tuple): Two element tuple used to preprocess the input.
            First substract the first element, then divide the second element.
    """

    def __init__(self, bounds, channel_axis, preprocess=None):
        assert len(bounds) == 2
        assert channel_axis in [0, 1, 2, 3]

        self._bounds = bounds
        self._channel_axis = channel_axis

        # Make self._preprocess to be (0,1) if possible, so that don't need
        # to do substract or divide.
        if preprocess is not None:
            sub, div = np.array(preprocess)
            if not np.any(sub):
                sub = 0
            if np.all(div == 1):
                div = 1
            assert (div is None) or np.all(div)
            self._preprocess = (sub, div)
        else:
            self._preprocess = (0, 1)

    def bounds(self):
        """
        Return the upper and lower bounds of the model.
        """
        return self._bounds

    def channel_axis(self):
        """
        Return the channel axis of the model.
        """
        return self._channel_axis

    def _process_input(self, input_):
        res = None
        sub, div = self._preprocess
        if np.any(sub != 0):
            res = input_ - sub
        if not np.all(sub == 1):
            if res is None:  # "res = input_ - sub" is not executed!
                res = old_div(input_, div)
            else:
                res /= div
        if res is None:  # "res = (input_ - sub)/ div" is not executed!
            return input_
        return res

    @abstractmethod
    def predict(self, data):
        """
        Calculate the prediction of the data.

        Args:
            data(numpy.ndarray): input data with shape (size,
            height, width, channels).

        Return:
            numpy.ndarray: predictions of the data with shape (batch_size,
                num_of_classes).
        """
        raise NotImplementedError

    @abstractmethod
    def num_classes(self):
        """
        Determine the number of the classes

        Return:
            int: the number of the classes
        """
        raise NotImplementedError

    @abstractmethod
    def gradient(self, data, label):
        """
        Calculate the gradient of the cross-entropy loss w.r.t the image.

        Args:
            data(numpy.ndarray): input data with shape (size, height, width,
            channels).
            label(int): Label used to calculate the gradient.

        Return:
            numpy.ndarray: gradient of the cross-entropy loss w.r.t the image
                with the shape (height, width, channel).
        """
        raise NotImplementedError

    @abstractmethod
    def predict_name(self):
        """
        Get the predict name, such as "softmax",etc.
        :return: string
        """
        raise NotImplementedError
