import unittest
import os
import torch
from torch.utils.data import DataLoader
import albumentations as A
from utils import dataloader
from utils.dataset import DatasetAugPil
from app.config import Mode, Config
config = Config()

TEST_DATASET_DIR = "./dataset/"
TEST_DATASET_CNT_TRAIN = 6
TEST_DATASET_CNT_EVAL = 3
TEST_DATASET_CNT_TEST = 3


class MyLoaderTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        dir = os.path.dirname(__file__)
        self.test_path = os.path.join(dir, TEST_DATASET_DIR)
        self.instance_train = dataloader.MyLoader(Mode.TRAIN, self.test_path)
        self.instance_eval = dataloader.MyLoader(Mode.EVAL, self.test_path)
        self.instance_test = dataloader.MyLoader(Mode.TEST, self.test_path)
        self.all_txt_path_train = os.path.join(
            self.instance_train.dataset_path, config.image_list_txt_name)
        self.all_txt_path_eval = os.path.join(
            self.instance_eval.dataset_path, config.image_list_txt_name)
        self.all_txt_path_test = os.path.join(
            self.instance_test.dataset_path, config.image_list_txt_name)

    def test_1_init_train(self):
        self.assertIs(self.instance_train.mode, Mode.TRAIN)
        path = os.path.join(self.test_path, Mode.TRAIN.str())
        self.assertEqual(self.instance_train.dataset_path, path)
        self.assertIsInstance(self.instance_train.tf, A.Compose)

    def test_2_init_eval(self):
        self.assertIs(self.instance_eval.mode, Mode.EVAL)
        path = os.path.join(self.test_path, Mode.EVAL.str())
        self.assertEqual(self.instance_eval.dataset_path, path)
        self.assertIsInstance(self.instance_eval.tf, A.Compose)

    def test_3_init_test(self):
        self.assertIs(self.instance_test.mode, Mode.TEST)
        path = os.path.join(self.test_path, Mode.TEST.str())
        self.assertEqual(self.instance_test.dataset_path, path)
        self.assertIsInstance(self.instance_test.tf, A.Compose)

    def test_4_create_images_filenames_list_txt_train(self):
        self.instance_train._create_images_filenames_list_txt()
        self.assertTrue(os.path.exists(self.all_txt_path_train))
        with open(self.all_txt_path_train) as f:
            path = f.read().splitlines()
        dname = os.path.join(
            self.instance_train.dataset_path, config.image_dir_name)
        for rt, _, files in os.walk(dname):
            for f in files:
                self.assertIn(os.path.join(rt, f), path)

    def test_5_create_images_filenames_list_txt_eval(self):
        self.instance_eval._create_images_filenames_list_txt()
        self.assertTrue(os.path.exists(self.all_txt_path_eval))
        with open(self.all_txt_path_eval) as f:
            path = f.read().splitlines()
        dname = os.path.join(
            self.instance_eval.dataset_path, config.image_dir_name)
        for rt, _, files in os.walk(dname):
            for f in files:
                self.assertIn(os.path.join(rt, f), path)

    def test_6_create_images_filenames_list_txt_test(self):
        self.instance_test._create_images_filenames_list_txt()
        self.assertTrue(os.path.exists(self.all_txt_path_test))
        with open(self.all_txt_path_test) as f:
            path = f.read().splitlines()
        dname = os.path.join(
            self.instance_test.dataset_path, config.image_dir_name)
        for rt, _, files in os.walk(dname):
            for f in files:
                self.assertIn(os.path.join(rt, f), path)

    def test_7_get_image_filenames_train(self):
        paths = self.instance_train._get_image_filenames()
        self.assertEqual(len(paths), TEST_DATASET_CNT_TRAIN)

    def test_8_get_image_filenames_eval(self):
        paths = self.instance_eval._get_image_filenames()
        self.assertEqual(len(paths), TEST_DATASET_CNT_EVAL)

    def test_9_get_image_filenames_test(self):
        paths = self.instance_test._get_image_filenames()
        self.assertEqual(len(paths), TEST_DATASET_CNT_TEST)

    def test_10_get_dataset_train(self):
        dataset = self.instance_train._get_dataset()
        self.assertIsInstance(dataset, DatasetAugPil)
        self.assertEqual(len(dataset.file_paths), TEST_DATASET_CNT_TRAIN)

    def test_11_get_dataset_eval(self):
        dataset = self.instance_eval._get_dataset()
        self.assertIsInstance(dataset, DatasetAugPil)
        self.assertEqual(len(dataset.file_paths), TEST_DATASET_CNT_EVAL)

    def test_12_get_dataset_test(self):
        dataset = self.instance_test._get_dataset()
        self.assertIsInstance(dataset, DatasetAugPil)
        self.assertEqual(len(dataset.file_paths), TEST_DATASET_CNT_TEST)

    def test_13_get_loader_for_train(self):
        loader = self.instance_train._get_loader_for_train()
        self.assertIsInstance(loader, DataLoader)

    def test_14_get_loader_for_eval(self):
        loader = self.instance_eval._get_loader_for_eval()
        self.assertIsInstance(loader, DataLoader)

    def test_15_get_torch_dataloader_train(self):
        loader = self.instance_train.get_torch_dataloader()
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, config.batch_train)

    def test_16_get_torch_dataloader_eval(self):
        loader = self.instance_eval.get_torch_dataloader()
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, config.batch_eval)

    def test_16_get_torch_dataloader_test(self):
        loader = self.instance_test.get_torch_dataloader()
        self.assertIsInstance(loader, DataLoader)
        self.assertEqual(loader.batch_size, config.batch_eval)

    def test_17_load(self):
        loader = self.instance_eval.get_torch_dataloader()
        image = next(iter(loader))
        self.assertIsInstance(image, torch.Tensor)
        n, c, h, w = image.shape  # batch_size, channel, height, width
        self.assertEqual(n, config.batch_eval)
        self.assertEqual(c, config.input.c)
        self.assertEqual(h, config.input.h)
        self.assertEqual(w, config.input.w)

    def tearDown(self) -> None:
        try:
            os.remove(self.all_txt_path_train)
        except OSError:
            pass

        try:
            os.remove(self.all_txt_path_eval)
        except OSError:
            pass

        try:
            os.remove(self.all_txt_path_test)
        except OSError:
            pass

        super().tearDown()


if __name__ == '__main__':
    unittest.main()
