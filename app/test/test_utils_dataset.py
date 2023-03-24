import unittest
import os
import torch
from utils import dataset
from utils.augment import MyTransform
from app.config import Mode

TEST_IMAGE_FILENAME = "./dataset/train/images/placekitten_640x360_0.jpg"
TEST_FILE_PATHS = [os.path.join(os.path.dirname(__file__), TEST_IMAGE_FILENAME),
                   'filename1', 'filename2', 'filename3']
TEST_LABELS = ["label1", "label2", "label3", "label4"]


class DatasetAugPilTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.file_paths = TEST_FILE_PATHS
        self.tf = MyTransform(Mode.TRAIN).get()
        self.instance = dataset.DatasetAugPil(self.file_paths, self.tf)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        self.assertIsInstance(self.instance.__getitem__(0), torch.Tensor)


class DatasetAugPilLabelTestCase(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self.file_paths = TEST_FILE_PATHS
        self.labels = TEST_LABELS
        self.tf = MyTransform(Mode.TRAIN).get()
        self.instance = dataset.DatasetAugPilLabel(
            self.file_paths, self.labels, self.tf)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)
        self.assertCountEqual(self.labels, self.instance.labels)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        image, label = self.instance.__getitem__(0)
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, str)


if __name__ == '__main__':
    unittest.main()
