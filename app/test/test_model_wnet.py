import pytest
import torch
from app.model import wnet

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
def Wnet_outputs(input_tensor):
    m = wnet.Wnet(in_channels=TEST_CHAN_1,
                  out_channels=TEST_CHAN_1, num_segment=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor)
    m.train()
    return outputs


def test_wnet_shape(Wnet_outputs):
    assert Wnet_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_1, TEST_SIZE, TEST_SIZE])


def test_wnet_compute_grad(Wnet_outputs, input_tensor):
    # Mask loss for certain samples in batch
    batch_size = Wnet_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(Wnet_outputs)
    mask[mask_idx] = 0
    Wnet_outputs = Wnet_outputs * mask

    # Compute backward pass
    loss = Wnet_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)
