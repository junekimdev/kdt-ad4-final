import torch
import torch.nn as nn
import torch.optim as optim
from app.model.wnet import Wnet
from app.loader.dataloader import MyLoader
from app.config import Config, Mode
config = Config()


class Trainer:
    def __init__(self, dataset_root, Model=Wnet, Loader=MyLoader) -> None:
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        print(f"PyTorch uses [{self.device}]")

        self.dataloader = Loader(Mode.TRAIN, dataset_root)
        self.model = Model().to(self.device).train()
        self.encoder_loss = nn.NLLLoss().to(self.device)  # Negative Log Likelihood Loss
        self.decoder_loss = nn.MSELoss().to(self.device)  # Mean Squared Error Loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr_init)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.lr_decay_step, gamma=config.lr_decay_amount)

    def run(self):
        # TODO
        pass
