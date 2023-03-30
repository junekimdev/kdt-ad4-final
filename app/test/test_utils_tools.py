import pytest
import os
import torch
from app.utils import tools

TEST_BATCH = 1
TEST_CHAN = 3
TEST_SIZE = 224
TEST_NAME = "test_image.jpg"


@pytest.fixture
def output_dir():
    return os.path.join(os.path.dirname(__file__), "output")


@pytest.fixture
def input_tensor():
    tensor = torch.randn(TEST_BATCH, TEST_CHAN, TEST_SIZE, TEST_SIZE)
    tensor.requires_grad = True
    return tensor


def test_save_image(input_tensor, output_dir):
    try:
        tools.save_image(input_tensor, output_dir, TEST_NAME)
    except Exception as e:
        pytest.fail(str(e))
    else:
        for rt, _, files in os.walk(output_dir):
            for f in files:
                fname = os.path.join(rt, f)
                os.remove(fname)
