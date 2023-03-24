import pytest
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from app.utils.dataloader import MyLoader
from app.utils.dataset import DatasetAugPil
from app.config import Mode, Config
config = Config()

TEST_DATASET_DIR_NAME = "dataset"
TEST_DATASET_CNT_TRAIN = 6
TEST_DATASET_CNT_EVAL = 3
TEST_DATASET_CNT_TEST = 3


@pytest.fixture
def dataset_dir():
    dir = os.path.dirname(__file__)
    return os.path.join(dir, TEST_DATASET_DIR_NAME)


@pytest.fixture
def all_txt_path_train(dataset_dir):
    dataset_path = os.path.join(dataset_dir, Mode.TRAIN.str())
    all_txt_path = os.path.join(dataset_path, config.image_list_txt_name)
    yield all_txt_path

    # tear down
    try:
        os.remove(all_txt_path)
    except OSError:
        pass


def test_MyLoader_train(dataset_dir, all_txt_path_train):
    instance = MyLoader(Mode.TRAIN, dataset_dir)
    assert instance.mode is Mode.TRAIN
    assert isinstance(instance.tf, A.Compose)

    dataset_path = os.path.join(dataset_dir, Mode.TRAIN.str())
    assert instance.dataset_path == dataset_path

    instance._create_images_filenames_list_txt()
    assert os.path.exists(all_txt_path_train)

    with open(all_txt_path_train) as f:
        path = f.read().splitlines()
    dname = os.path.join(instance.dataset_path, config.image_dir_name)
    for rt, _, files in os.walk(dname):
        for f in files:
            assert os.path.join(rt, f) in path

    file_paths = instance._get_image_filenames()
    assert len(file_paths) == TEST_DATASET_CNT_TRAIN

    dataset = instance._get_dataset()
    assert isinstance(dataset, DatasetAugPil)
    assert len(dataset.file_paths) == TEST_DATASET_CNT_TRAIN

    loader = instance._get_loader_for_train()
    assert isinstance(loader, DataLoader)

    torch_loader = instance.get_torch_dataloader()
    assert isinstance(torch_loader, DataLoader)
    assert torch_loader.batch_size == config.batch_train

    for image in torch_loader:
        assert isinstance(image, torch.Tensor)


@pytest.fixture
def all_txt_path_eval(dataset_dir):
    dataset_path = os.path.join(dataset_dir, Mode.EVAL.str())
    all_txt_path = os.path.join(dataset_path, config.image_list_txt_name)
    yield all_txt_path

    # tear down
    try:
        os.remove(all_txt_path)
    except OSError:
        pass


def test_MyLoader_eval(dataset_dir, all_txt_path_eval):
    instance = MyLoader(Mode.EVAL, dataset_dir)
    assert instance.mode is Mode.EVAL
    assert isinstance(instance.tf, A.Compose)

    dataset_path = os.path.join(dataset_dir, Mode.EVAL.str())
    assert instance.dataset_path == dataset_path

    instance._create_images_filenames_list_txt()
    assert os.path.exists(all_txt_path_eval)

    with open(all_txt_path_eval) as f:
        path = f.read().splitlines()
    dname = os.path.join(instance.dataset_path, config.image_dir_name)
    for rt, _, files in os.walk(dname):
        for f in files:
            assert os.path.join(rt, f) in path

    file_paths = instance._get_image_filenames()
    assert len(file_paths) == TEST_DATASET_CNT_EVAL

    dataset = instance._get_dataset()
    assert isinstance(dataset, DatasetAugPil)
    assert len(dataset.file_paths) == TEST_DATASET_CNT_EVAL

    loader = instance._get_loader_for_eval()
    assert isinstance(loader, DataLoader)

    torch_loader = instance.get_torch_dataloader()
    assert isinstance(torch_loader, DataLoader)
    assert torch_loader.batch_size == config.batch_eval

    for image in torch_loader:
        n, c, h, w = image.shape  # batch_size, channel, height, width
        assert isinstance(image, torch.Tensor)
        assert n == config.batch_eval
        assert c == config.input.c
        assert h == config.input.h
        assert w == config.input.w


@pytest.fixture
def all_txt_path_test(dataset_dir):
    dataset_path = os.path.join(dataset_dir, Mode.TEST.str())
    all_txt_path = os.path.join(dataset_path, config.image_list_txt_name)
    yield all_txt_path

    # tear down
    try:
        os.remove(all_txt_path)
    except OSError:
        pass


def test_MyLoader_test(dataset_dir, all_txt_path_test):
    instance = MyLoader(Mode.TEST, dataset_dir)
    assert instance.mode is Mode.TEST
    assert isinstance(instance.tf, A.Compose)

    dataset_path = os.path.join(dataset_dir, Mode.TEST.str())
    assert instance.dataset_path == dataset_path

    instance._create_images_filenames_list_txt()
    assert os.path.exists(all_txt_path_test)

    with open(all_txt_path_test) as f:
        path = f.read().splitlines()
    dname = os.path.join(instance.dataset_path, config.image_dir_name)
    for rt, _, files in os.walk(dname):
        for f in files:
            assert os.path.join(rt, f) in path

    file_paths = instance._get_image_filenames()
    assert len(file_paths) == TEST_DATASET_CNT_TEST

    dataset = instance._get_dataset()
    assert isinstance(dataset, DatasetAugPil)
    assert len(dataset.file_paths) == TEST_DATASET_CNT_TEST

    loader = instance._get_loader_for_eval()
    assert isinstance(loader, DataLoader)

    torch_loader = instance.get_torch_dataloader()
    assert isinstance(torch_loader, DataLoader)
    assert torch_loader.batch_size == config.batch_eval

    image = next(iter(torch_loader))
    assert isinstance(image, torch.Tensor)

    for image in torch_loader:
        n, c, h, w = image.shape  # batch_size, channel, height, width
        assert isinstance(image, torch.Tensor)
        assert n == config.batch_eval
        assert c == config.input.c
        assert h == config.input.h
        assert w == config.input.w
