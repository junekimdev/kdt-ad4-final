import os
import multiprocessing
from torch.utils.data import DataLoader
from utils.dataset import DatasetAugPil
from utils.augment import MyTransform
from app.config import Config, Mode
config = Config()


class MyLoader:
    def __init__(self, mode: Mode, dataset_root: str) -> None:
        self.mode = mode
        self.dataset_path = os.path.join(dataset_root, self.mode.str())
        self.tf = MyTransform(mode).get()

    def _create_images_filenames_list_txt(self) -> None:
        fname = os.path.join(self.dataset_path, config.image_list_txt_name)
        dname = os.path.join(self.dataset_path, config.image_dir_name)
        paths = [os.path.join(rt, f)+"\n"
                 for rt, _, files in os.walk(dname) for f in files]
        with open(fname, "w") as f:
            f.writelines(paths)

    def _get_image_filenames(self) -> list[str]:
        fname = os.path.join(self.dataset_path, config.image_list_txt_name)
        if not os.path.exists(fname):
            self._create_images_filenames_list_txt()
        paths = []
        with open(fname, "r") as f:
            paths = f.read().splitlines()
        return paths

    def _get_dataset(self) -> DatasetAugPil:
        return DatasetAugPil(self._get_image_filenames(), self.tf)

    def _get_loader_for_train(self) -> DataLoader:
        return DataLoader(dataset=self._get_dataset(),
                          batch_size=config.batch_train,
                          pin_memory=True,
                          drop_last=True,
                          shuffle=True,
                          num_workers=multiprocessing.cpu_count())

    def _get_loader_for_eval(self) -> DataLoader:
        return DataLoader(dataset=self._get_dataset(),
                          batch_size=config.batch_eval,
                          pin_memory=True,
                          drop_last=False,
                          shuffle=False,
                          num_workers=multiprocessing.cpu_count())

    def get_torch_dataloader(self) -> DataLoader:
        if self.mode is Mode.TRAIN:
            return self._get_loader_for_train()
        else:
            return self._get_loader_for_eval()
