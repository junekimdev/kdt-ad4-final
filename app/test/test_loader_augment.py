import pytest
import albumentations as A
from app.loader.augment import MyTransform
from app.config import Mode


def test_tf_train():
    tf = MyTransform(Mode.TRAIN)

    assert tf.mode is Mode.TRAIN
    assert isinstance(tf._get_tf_train(), A.Compose)
    assert isinstance(tf._get_tf_eval(), A.Compose)
    assert isinstance(tf.get(), A.Compose)


def test_tf_eval():
    tf = MyTransform(Mode.EVAL)

    assert tf.mode is Mode.EVAL
    assert isinstance(tf._get_tf_train(), A.Compose)
    assert isinstance(tf._get_tf_eval(), A.Compose)
    assert isinstance(tf.get(), A.Compose)


def test_tf_test():
    tf = MyTransform(Mode.TEST)

    assert tf.mode is Mode.TEST
    assert isinstance(tf._get_tf_train(), A.Compose)
    assert isinstance(tf._get_tf_eval(), A.Compose)
    assert isinstance(tf.get(), A.Compose)
