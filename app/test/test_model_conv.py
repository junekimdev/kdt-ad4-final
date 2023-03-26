import pytest
import torch
from app.model import conv

TEST_BATCH = 4
TEST_CHAN_1 = 3
TEST_CHAN_2 = 6
TEST_SIZE = 256


@pytest.fixture
def input_tensor():
    tensor = torch.randn(TEST_BATCH, TEST_CHAN_1, TEST_SIZE, TEST_SIZE)
    tensor.requires_grad = True
    return tensor


@pytest.fixture
def RegularConv_outputs(input_tensor):
    m = conv.RegularConv(in_channels=TEST_CHAN_1, out_channels=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor)
    m.train()
    return outputs


def test_conv_shape(RegularConv_outputs):
    assert RegularConv_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_2, TEST_SIZE, TEST_SIZE])


def test_conv_compute_grad(RegularConv_outputs, input_tensor):
    # Mask loss for certain samples in batch
    batch_size = RegularConv_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(RegularConv_outputs)
    mask[mask_idx] = 0
    RegularConv_outputs = RegularConv_outputs * mask

    # Compute backward pass
    loss = RegularConv_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


@pytest.fixture
def SeparableConv_outputs(input_tensor):
    m = conv.SeparableConv(in_channels=TEST_CHAN_1, out_channels=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor)
    m.train()
    return outputs


def test_separable_conv_shape(SeparableConv_outputs):
    assert SeparableConv_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_2, TEST_SIZE, TEST_SIZE])


def test_separable_conv_compute_grad(SeparableConv_outputs, input_tensor):
    # Mask loss for certain samples in batch
    batch_size = SeparableConv_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(SeparableConv_outputs)
    mask[mask_idx] = 0
    SeparableConv_outputs = SeparableConv_outputs * mask

    # Compute backward pass
    loss = SeparableConv_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)
