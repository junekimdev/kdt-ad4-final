import torch.nn as nn
from app.model.unet import Unet
from app.config import Config
config = Config()


class Wnet(nn.Module):
    def __init__(self, in_channels=config.input.c, out_channels=config.output.c, num_segment=config.K):
        super().__init__()
        self.encoder = Unet(in_channels, num_segment)
        self.softmax = nn.Softmax(dim=1)  # dim=1 is the channel
        self.decoder = Unet(num_segment, out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.softmax(x)
        x = self.decoder(x)
        return x
