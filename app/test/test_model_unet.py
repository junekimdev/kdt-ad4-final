import pytest
import torch
from app.model import unet

TEST_BATCH = 4
TEST_CHAN_1 = 3
TEST_CHAN_2 = 6
TEST_SIZE = 256


@pytest.fixture
def input_tensor_up():
    tensor = torch.randn(TEST_BATCH, TEST_CHAN_1, TEST_SIZE, TEST_SIZE)
    tensor.requires_grad = True
    return tensor


@pytest.fixture
def EncoderBlockReg_outputs(input_tensor_up):
    m = unet.EncoderBlockReg(in_channels=TEST_CHAN_1, out_channels=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    _, outputs = m(input_tensor_up)
    m.train()
    return outputs


def test_encoder_reg_shape(EncoderBlockReg_outputs):
    assert EncoderBlockReg_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_2, TEST_SIZE//2, TEST_SIZE//2])


def test_encoder_reg_compute_grad(EncoderBlockReg_outputs, input_tensor_up):
    # Mask loss for certain samples in batch
    batch_size = EncoderBlockReg_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(EncoderBlockReg_outputs)
    mask[mask_idx] = 0
    EncoderBlockReg_outputs = EncoderBlockReg_outputs * mask

    # Compute backward pass
    loss = EncoderBlockReg_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor_up.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


@pytest.fixture
def EncoderBlockSep_outputs(input_tensor_up):
    m = unet.EncoderBlockSep(in_channels=TEST_CHAN_1, out_channels=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    _, outputs = m(input_tensor_up)
    m.train()
    return outputs


def test_encoder_sep_shape(EncoderBlockSep_outputs):
    assert EncoderBlockSep_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_2, TEST_SIZE//2, TEST_SIZE//2])


def test_encoder_sep_compute_grad(EncoderBlockSep_outputs, input_tensor_up):
    # Mask loss for certain samples in batch
    batch_size = EncoderBlockSep_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(EncoderBlockSep_outputs)
    mask[mask_idx] = 0
    EncoderBlockSep_outputs = EncoderBlockSep_outputs * mask

    # Compute backward pass
    loss = EncoderBlockSep_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor_up.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


@pytest.fixture
def input_tensor_down():
    tensor = torch.randn(TEST_BATCH, TEST_CHAN_2, TEST_SIZE, TEST_SIZE)
    tensor.requires_grad = True
    return tensor


@pytest.fixture
def skip_tensor():
    return torch.randn(TEST_BATCH, TEST_CHAN_1, TEST_SIZE*2, TEST_SIZE*2)


@pytest.fixture
def DecoderBlockReg_outputs(input_tensor_down, skip_tensor):
    m = unet.DecoderBlockReg(in_channels=TEST_CHAN_2, out_channels=TEST_CHAN_1)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor_down, skip_tensor)
    m.train()
    return outputs


def test_decoder_reg_shape(DecoderBlockReg_outputs):
    assert DecoderBlockReg_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_1, TEST_SIZE*2, TEST_SIZE*2])


def test_decoder_reg_compute_grad(DecoderBlockReg_outputs, input_tensor_down):
    # Mask loss for certain samples in batch
    batch_size = DecoderBlockReg_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(DecoderBlockReg_outputs)
    mask[mask_idx] = 0
    DecoderBlockReg_outputs = DecoderBlockReg_outputs * mask

    # Compute backward pass
    loss = DecoderBlockReg_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor_down.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


@pytest.fixture
def DecoderBlockSep_outputs(input_tensor_down, skip_tensor):
    m = unet.DecoderBlockSep(in_channels=TEST_CHAN_2, out_channels=TEST_CHAN_1)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor_down, skip_tensor)
    m.train()
    return outputs


def test_decoder_sep_shape(DecoderBlockSep_outputs):
    assert DecoderBlockSep_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_1, TEST_SIZE*2, TEST_SIZE*2])


def test_decoder_sep_compute_grad(DecoderBlockSep_outputs, input_tensor_down):
    # Mask loss for certain samples in batch
    batch_size = DecoderBlockSep_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(DecoderBlockSep_outputs)
    mask[mask_idx] = 0
    DecoderBlockSep_outputs = DecoderBlockSep_outputs * mask

    # Compute backward pass
    loss = DecoderBlockSep_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor_down.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)


@pytest.fixture
def Unet_outputs(input_tensor_up):
    m = unet.Unet(in_channels=TEST_CHAN_1, out_channels=TEST_CHAN_2)
    m.eval()  # deactivate batch norm
    outputs = m(input_tensor_up)
    m.train()
    return outputs


def test_unet_shape(Unet_outputs):
    assert Unet_outputs.shape == torch.Size(
        [TEST_BATCH, TEST_CHAN_2, TEST_SIZE, TEST_SIZE])


def test_unet_compute_grad(Unet_outputs, input_tensor_up):
    # Mask loss for certain samples in batch
    batch_size = Unet_outputs.shape[0]
    mask_idx = torch.randint(0, batch_size, ())
    mask = torch.ones_like(Unet_outputs)
    mask[mask_idx] = 0
    Unet_outputs = Unet_outputs * mask

    # Compute backward pass
    loss = Unet_outputs.mean()
    loss.backward()

    # Check if gradient exists and is zero for masked samples
    for i, grad in enumerate(input_tensor_up.grad):
        if i == mask_idx:
            assert torch.all(grad == 0).item()
        else:
            assert not torch.all(grad == 0)
