"""
Defines a class that contains the original object, the target and the
adversarial example.

"""
from builtins import object
import numpy as np
import  logging
logger=logging.getLogger(__name__)

class Adversary(object):
    """
    Adversary contains the original object, the target and the adversarial
    example.
    """

    def __init__(self, original, original_label=None):
        """
        :param original: The original instance, such as an image.
        :param original_label: The original instance's label.
        """
        assert original is not None

        self.original_label = original_label
        # 定向攻击的目标
        self.target_label = None
        self.adversarial_label = None
        # 保存原始数据 强制拷贝
        self.__original = np.copy(original)
        self.__target = None
        self.__is_targeted_attack = False
        # 保存生成的对抗样本
        self.__adversarial_example = None
        self.__bad_adversarial_example = None

    def set_target(self, is_targeted_attack, target=None, target_label=None):
        """
        Set the target be targeted or untargeted.

        :param is_targeted_attack: bool
        :param target: The target.
        :param target_label: If is_targeted_attack is true and target_label is
                    None, self.target_label will be set by the Attack class.
                    If is_targeted_attack is false, target_label must be None.
        """
        assert (target_label is None) or is_targeted_attack
        self.__is_targeted_attack = is_targeted_attack
        self.target_label = target_label
        self.__target = target
        if not is_targeted_attack:
            self.target_label = None
            self.__target = None

    def set_original(self, original, original_label=None):
        """
        Reset the original.

        :param original: Original instance.
        :param original_label: Original instance's label.
        """
        if original != self.__original:
            self.__original = original
            self.original_label = original_label
            self.__adversarial_example = None
            self.__bad_adversarial_example = None
        if original is None:
            self.original_label = None

    def _is_successful(self, adversarial_label):
        """
        Is the adversarial_label is the expected adversarial label.

        :param adversarial_label: adversarial label.
        :return: bool
        """
        if self.target_label is not None:
            return adversarial_label == self.target_label
        else:
            return (adversarial_label is not None) and \
                   (adversarial_label != self.original_label)

    def is_successful(self):
        """
        Has the adversarial example been found.

        :return: bool
        """
        return self._is_successful(self.adversarial_label)

    def try_accept_the_example(self, adversarial_example, adversarial_label):
        """
        If adversarial_label the target label that we are finding.
        The adversarial_example and adversarial_label will be accepted and
        True will be returned.

        :return: bool
        """
        assert adversarial_example is not None
        assert self.__original.shape == adversarial_example.shape

        ok = self._is_successful(adversarial_label)
        if ok:
            self.__adversarial_example = np.copy(adversarial_example)
            self.adversarial_label = adversarial_label
        else:
            self.__bad_adversarial_example = np.copy(adversarial_example)
        return ok

    def perturbation(self, multiplying_factor=1.0):
        """
        The perturbation that the adversarial_example is added.

        :param multiplying_factor: float.
        :return: The perturbation that is multiplied by multiplying_factor.
        """
        assert self.__original is not None
        assert (self.__adversarial_example is not None) or \
               (self.__bad_adversarial_example is not None)
        if self.__adversarial_example is not None:
            return multiplying_factor * (
                self.__adversarial_example - self.__original)
        else:
            return multiplying_factor * (
                self.__bad_adversarial_example - self.__original)

    @property
    def is_targeted_attack(self):
        """
        :property: is_targeted_attack
        """
        return self.__is_targeted_attack

    @property
    def target(self):
        """
        :property: target
        """
        return self.__target

    @property
    def original(self):
        """
        :property: original
        """
        return self.__original

    @property
    def adversarial_example(self):
        """
        :property: adversarial_example
        """
        return self.__adversarial_example

    @property
    def bad_adversarial_example(self):
        """
        :property: bad_adversarial_example
        """
        return self.__bad_adversarial_example
