import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from app.model.wnet import Wnet
from app.loader.dataloader import MyLoader
from app.run.runner import Runnable
from app.utils.tools import save_image, save_clusters
from app.config import Config, Mode
config = Config()


class Evaluator(Runnable):
    global_image_id = 0  # class attribute

    @classmethod
    def load(cls, checkpoint_path: str, dataset_root: str, output_dir: str):
        self = cls(dataset_root, output_dir)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model"])

        self._send_to_device()
        return self

    def __init__(self, dataset_root: str, output_dir: str, model=None, loader=None,  K=config.K):
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        print(f"Evaluator uses [{self.device}]")

        self.K = K
        self.dataset_root = os.path.abspath(dataset_root)
        self.output_dir = os.path.abspath(output_dir)
        self.writer = SummaryWriter(self.output_dir)
        print(f"Evaluator outputs results to [{self.output_dir}]")

        self.dataloader = MyLoader(Mode.EVAL, self.dataset_root).torch() \
            if loader is None else loader
        self.model = Wnet() if model is None else model

        self._send_to_device()

    def _send_to_device(self):
        self.model.to(self.device)

    def run(self) -> None:
        self.model.eval()
        summary(self.model, config.input.shape, config.batch_eval)
        print("Start evaluating...")

        start_at = time.time()
        for batch in self.dataloader:
            if batch is None:  # Invalid batch
                continue

            t = time.time()
            # Encoder
            inference = self._infer(batch, run_decoder=False)
            filename = f"eval-{t}-u1"
            save_clusters(inference, self.K, self.output_dir, filename)

            # Encoder+Decoder
            inference = self._infer(batch, run_decoder=True)
            filename = f"eval-{t}-u2"
            save_image(inference, self.output_dir, filename)

        dt = time.time()-start_at
        print(f"It took {dt:.3f} sec to evaluate")

    def _infer(self, batch: torch.Tensor, run_decoder=True) -> torch.Tensor:
        img = batch  # Unpack if batch has labels
        img = img.to(self.device)
        with torch.no_grad():
            inference = self.model(img, run_decoder=run_decoder)
        return inference
