import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from PIL import Image
from app.model.wnet import Wnet
from app.loader.dataloader import MyLoader
from app.loss.soft_ncut import SoftNCutLoss
from app.config import Config, Mode
config = Config()


class Trainer:
    @classmethod
    def load(cls, checkpoint_path, dataset_root, output_dir):
        device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        instance = Trainer(dataset_root, output_dir)
        instance.model.load_state_dict(checkpoint["model"])
        instance.optimizer.load_state_dict(checkpoint["optimizer"])
        instance.scheduler.load_state_dict(checkpoint["scheduler"])
        instance.encoder_loss.load_state_dict(checkpoint["encoder_loss"])
        instance.decoder_loss.load_state_dict(checkpoint["decoder_loss"])
        instance.epoch = checkpoint["epoch"]
        instance.iter = checkpoint["iter"]
        instance._send_to_device()
        return instance

    def __init__(self, dataset_root, output_dir,
                 model=None, loader=None, optimizer=None, scheduler=None,
                 encoder_loss=None, decoder_loss=None, epoch=0, iter=0):
        self.epoch = epoch
        self.iter = iter
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        print(f"PyTorch uses [{self.device}]")

        self.output_dir = output_dir
        self.writer = SummaryWriter(output_dir)
        print(f"PyTorch outputs results to [{output_dir}]")

        self.dataloader = MyLoader(Mode.TRAIN, dataset_root) \
            if loader is None else loader
        self.model = Wnet() if model is None else model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr_init) \
            if optimizer is None else optimizer
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.lr_decay_epoch, gamma=config.lr_decay_amount) \
            if scheduler is None else scheduler
        # SoftNCutLoss derived from Negative Log Likelihood(NLL) Loss
        self.encoder_loss = nn.SoftNCutLoss() if encoder_loss is None else encoder_loss
        # Mean Squared Error(MSE) Loss
        self.decoder_loss = nn.MSELoss() if decoder_loss is None else decoder_loss
        self._send_to_device()

    def _send_to_device(self):
        self.model.to(self.device)
        self.encoder_loss.to(self.device)
        self.decoder_loss.to(self.device)

    def run(self):
        self.model.state_dict()
        self.model.train()
        while True:
            self.run_epoch()

    def run_epoch(self, start_at=time.time()):
        self.epoch += 1
        for batch in self.dataloader:
            if batch is None:  # Invalid batch
                continue
            self.run_iter(batch)

        self.scheduler.step()
        self.writer.add_scalar("Time/epoch", time.time()-start_at, self.epoch)

        if not self.epoch % config.save_period_epoch:
            self.save()

    def run_iter(self, batch, start_at=time.time()):
        self.iter += 1
        img = batch  # Unpack if batch has labels
        img = img.to(self.device)

        # Phase 1 - Encoder Only Feedback
        self.optimizer.zero_grad()
        encoder_res = self.model(img, run_decoder=False)
        encoder_loss = self.encoder_loss(encoder_res, img)
        encoder_loss.backward()
        self.optimizer.step()

        # Phase 2 - Encoder+Decoder Feedback
        self.optimizer.zero_grad()
        decoder_res = self.model(img, run_decoder=True)
        decoder_loss = self.decoder_loss(decoder_res, img)
        decoder_loss.backward()
        self.optimizer.step()

        # Output
        if not self.iter % config.output_period_iter:
            self.write_iter(encoder_loss, decoder_loss, time.time()-start_at)

    def write_iter(self, encoder_loss, decoder_loss, time_spent):
        self.writer.add_scalar("Loss/Encoder", encoder_loss.item(), self.iter)
        self.writer.add_scalar("Loss/Decoder", decoder_loss.item(), self.iter)
        self.writer.add_scalar("LR", self.scheduler.get_last_lr(), self.iter)
        self.writer.add_scalar("Time/Iter", time_spent, self.iter)

    def save(self):
        path = os.path.join(self.output_dir, f"model_epoch{self.epoch}.pth")
        save_dict = {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict(),
                     "encoder_loss": self.encoder_loss.state_dict(),
                     "decoder_loss": self.decoder_loss.state_dict(),
                     "epoch": self.epoch,
                     "iter": self.iter}
        torch.save(save_dict, path)