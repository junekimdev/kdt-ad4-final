import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from app.model.wnet import Wnet
from app.loader.dataloader import MyLoader
from app.loss.soft_ncut import SoftNCutLoss
from app.run.runner import Runnable
from app.utils.tools import save_image
from app.config import Config, Mode
config = Config()


class Trainer(Runnable):
    @classmethod
    def load(cls, checkpoint_path: str, dataset_root: str, output_dir: str):
        self = cls(dataset_root, output_dir)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.scheduler.load_state_dict(checkpoint["scheduler"])
        self.encoder_loss.load_state_dict(checkpoint["encoder_loss"])
        self.decoder_loss.load_state_dict(checkpoint["decoder_loss"])
        self.epoch = checkpoint["epoch"]
        self.iter = checkpoint["iter"]

        self._send_to_device()
        return self

    def __init__(self, dataset_root: str, output_dir: str,
                 model=None, loader=None, optimizer=None, scheduler=None,
                 encoder_loss=None, decoder_loss=None, epoch=0, iter=0):
        self.epoch = epoch
        self.iter = iter
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        print(f"Trainer uses [{self.device}]")

        self.dataset_root = os.path.abspath(dataset_root)
        self.output_dir = os.path.abspath(output_dir)
        self.writer = SummaryWriter(self.output_dir)
        print(f"Trainer outputs results to [{self.output_dir}]")

        self.dataloader = MyLoader(Mode.TRAIN, self.dataset_root).torch() \
            if loader is None else loader
        self.model = Wnet() if model is None else model
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr_init) \
            if optimizer is None else optimizer
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.lr_decay_epoch, gamma=config.lr_decay_ratio) \
            if scheduler is None else scheduler
        # SoftNCutLoss derived from Negative Log Likelihood(NLL) Loss
        self.encoder_loss = SoftNCutLoss() if encoder_loss is None else encoder_loss
        # Mean Squared Error(MSE) Loss
        self.decoder_loss = nn.MSELoss() if decoder_loss is None else decoder_loss
        self._send_to_device()

    def _send_to_device(self) -> None:
        self.model.to(self.device)
        self.encoder_loss.to(self.device)
        self.decoder_loss.to(self.device)

    def run(self) -> None:
        summary(self.model, config.input.shape, config.batch_train)
        self.model.train()
        print("Start training...")
        while True:
            self._run_epoch()

    def _run_epoch(self) -> None:
        self.epoch += 1
        start_at = time.time()
        for batch in self.dataloader:
            if batch is None:  # Invalid batch
                continue
            self._run_iter(batch)

        self.scheduler.step()
        dt = time.time()-start_at
        self.writer.add_scalar("Time/Epoch", dt, self.epoch)
        print(f"{self.epoch} epoch | took {dt:.3f} sec for an epoch")

        # Save model
        if not self.epoch % config.save_period_epoch:
            # Save
            self._save()
            # Run eval
            self.model.eval()
            self._run_eval()
            self.model.train()

    def _run_iter(self, batch: torch.Tensor) -> None:
        self.iter += 1
        start_at = time.time()
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
            dt = time.time()-start_at
            self._write_iter(encoder_loss, decoder_loss, dt)
            print(f"{self.iter} iter | took {dt:.3f} sec for an iter")

    def _write_iter(self, encoder_loss: torch.Tensor, decoder_loss: torch.Tensor, time_spent: float) -> None:
        self.writer.add_scalar("Loss/Encoder", encoder_loss.item(), self.iter)
        self.writer.add_scalar("Loss/Decoder", decoder_loss.item(), self.iter)
        self.writer.add_scalar(
            "LR", self.scheduler.get_last_lr()[0], self.iter)
        self.writer.add_scalar("Time/Iter", time_spent, self.iter)

    def _save(self) -> None:
        path = os.path.join(self.output_dir, f"model_epoch{self.epoch}.pth")
        save_dict = {"model": self.model.state_dict(),
                     "optimizer": self.optimizer.state_dict(),
                     "scheduler": self.scheduler.state_dict(),
                     "encoder_loss": self.encoder_loss.state_dict(),
                     "decoder_loss": self.decoder_loss.state_dict(),
                     "epoch": self.epoch,
                     "iter": self.iter}
        torch.save(save_dict, path)
        print(f"Model has been saved after {self.epoch} epoch")

    def _run_eval(self):
        print("Start evaluating...")

        start_at = time.time()
        dataloader = MyLoader(Mode.EVAL, self.dataset_root).torch()
        infers = []
        for batch in dataloader:
            if batch is None:  # Invalid batch
                continue

            t = time.time()
            # Encoder
            inference = self._infer(batch, run_decoder=False)
            filename = f"eval-{t}-1.jpg"
            save_image(inference, self.output_dir, filename)
            infers.append(inference)

            # Encoder+Decoder
            inference = self._infer(batch, run_decoder=True)
            filename = f"eval-{t}-2.jpg"
            save_image(inference, self.output_dir, filename)
            infers.append(inference)

        # Output to tensorboard
        stacked_infers = torch.stack(infers)
        self.writer.add_images("Eval/Images", stacked_infers, self.epoch)

        dt = time.time()-start_at
        print(f"It took {dt:.3f} sec to evaluate")

    def _infer(self, batch: torch.Tensor, run_decoder=True) -> torch.Tensor:
        img = batch  # Unpack if batch has labels
        img = img.to(self.device)
        inference = self.model(img, run_decoder=run_decoder)
        return inference
