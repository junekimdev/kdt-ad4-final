import pytest
import os
import torch
from app.loader.dataset import DatasetAugPil, DatasetAugPilLabel
from app.loader.augment import MyTransform
from app.config import Mode


@pytest.fixture
def image_filename():
    return "./dataset/train/images/placekitten_640x360_0.jpg"


@pytest.fixture
def file_paths(image_filename):
    return [os.path.join(os.path.dirname(__file__), image_filename),
            'filename1', 'filename2', 'filename3']


@pytest.fixture
def labels():
    return ["label1", "label2", "label3", "label4"]


@pytest.fixture
def tf():
    return MyTransform(Mode.TRAIN).get()


def test_DatasetAugPil(file_paths, tf):
    instance = DatasetAugPil(file_paths, tf)

    assert len(instance.file_paths) == len(file_paths)
    assert len(instance) == len(file_paths)
    assert isinstance(instance.__getitem__(0), torch.Tensor)


def test_DatasetAugPilLabel(file_paths, labels, tf):
    instance = DatasetAugPilLabel(file_paths, labels, tf)
    image, label = instance.__getitem__(0)

    assert len(instance.file_paths) == len(file_paths)
    assert len(instance.labels) == len(labels)
    assert len(instance) == len(file_paths)
    assert isinstance(image, torch.Tensor)
    assert isinstance(label, str)
