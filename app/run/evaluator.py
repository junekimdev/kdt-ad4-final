import time
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import numpy as np
from PIL import Image
from app.model.wnet import Wnet
from app.loader.dataloader import MyLoader
from app.run.runner import Runnable
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

    def __init__(self, dataset_root: str, output_dir: str, model=None, loader=None):
        self.device = torch.device("cuda") \
            if torch.cuda.is_available() else torch.device("cpu")
        print(f"Evaluator uses [{self.device}]")

        self.output_dir = output_dir
        self.writer = SummaryWriter(output_dir)
        print(f"Evaluator outputs results to [{output_dir}]")

        self.dataloader = MyLoader(Mode.EVAL, dataset_root).torch() \
            if loader is None else loader
        self.model = Wnet() if model is None else model

        self._send_to_device()

    def _send_to_device(self):
        self.model.to(self.device)

    def run(self, *, print_summary=True, save_as_image=True, write_on_board=True) -> None:
        if print_summary:
            summary(self.model, config.input.shape, config.batch_eval)
        self.model.eval()
        print("Start evaluating...")

        start_at = time.time()
        infers = []
        for batch in self.dataloader:
            if batch is None:  # Invalid batch
                continue
            inference = self._run_iter(batch, write_on_board)
            if save_as_image:
                self._save_image(inference)
            if write_on_board:
                infers.append(inference)

        # Output to tensorboard
        if write_on_board:
            batch_infers = torch.stack(infers)
            self._write_on_board(batch_infers)

        dt = time.time()-start_at
        print(f"It took {dt:.3f} sec to evaluate")

    def _write_on_board(self, batch_infers: torch.Tensor):
        self.writer.add_images("Eval/Images", batch_infers,
                               Evaluator.global_image_id)
        Evaluator.global_image_id = + 1

    def _run_iter(self, batch: torch.Tensor) -> torch.Tensor:
        img = batch  # Unpack if batch has labels
        img = img.to(self.device)
        inference = self.model(img, run_decoder=True)
        return inference

    def _save_image(self, inference: torch.Tensor):
        # change shape to h,w,3
        img_hwc = inference.squeeze(0).transpose(1, 2, 0)
        assert img_hwc.shape[-1] == 3, "Tensor's channel is not 3"

        img_np = np.array(img_hwc, dtype=np.uint8)  # convert to numpy array
        img_res = Image.fromarray(img_np)  # convert to pillow image

        dname = os.path.join(self.output_dir, "images")
        try:
            os.makedirs(dname)
        except FileExistsError:
            pass
        t = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fname = os.path.join(dname, f"eval-{t}.jpg")
        img_res.save(fname)
        print(f"An image has been saved as {fname}")
