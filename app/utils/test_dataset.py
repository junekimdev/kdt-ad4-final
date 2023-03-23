import unittest
import os
from utils import dataset
from PIL import Image
import albumentations as A

TEST_IMAGE_FILENAME = "../test_image.jpg"


def get_test_image_path():
    return os.path.join(os.path.dirname(__file__), TEST_IMAGE_FILENAME)


class DatasetPilTestCase(unittest.TestCase):
    def setUp(self):
        self.file_paths = [get_test_image_path(),
                           'filename1', 'filename2', 'filename3']
        self.instance = dataset.DatasetPil(self.file_paths)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        self.assertIsInstance(self.instance.__getitem__(0), Image.Image)


class DatasetPilLabelTestCase(unittest.TestCase):
    def setUp(self):
        self.file_paths = [get_test_image_path(),
                           'filename1', 'filename2', 'filename3']
        self.labels = ["label1", "label2", "label3", "label4"]
        self.instance = dataset.DatasetPilLabel(self.file_paths, self.labels)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)
        self.assertCountEqual(self.labels, self.instance.labels)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        image, label = self.instance.__getitem__(0)
        self.assertIsInstance(image, Image.Image)
        self.assertIsInstance(label, str)


class DatasetAugPilTestCase(unittest.TestCase):
    def setUp(self):
        self.file_paths = [get_test_image_path(),
                           'filename1', 'filename2', 'filename3']
        self.tf = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
        ])
        self.instance = dataset.DatasetAugPil(self.file_paths, self.tf)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        self.assertIsInstance(self.instance.__getitem__(0), Image.Image)


class DatasetAugPilLabelTestCase(unittest.TestCase):
    def setUp(self):
        self.file_paths = [get_test_image_path(),
                           'filename1', 'filename2', 'filename3']
        self.labels = ["label1", "label2", "label3", "label4"]
        self.tf = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(),
        ])
        self.instance = dataset.DatasetAugPilLabel(
            self.file_paths, self.labels, self.tf)

    def test_1_init(self):
        self.assertCountEqual(self.file_paths, self.instance.file_paths)
        self.assertCountEqual(self.labels, self.instance.labels)

    def test_2_len(self):
        self.assertEqual(len(self.file_paths), len(self.instance))

    def test_3_get_item(self):
        image, label = self.instance.__getitem__(0)
        self.assertIsInstance(image, Image.Image)
        self.assertIsInstance(label, str)


if __name__ == '__main__':
    unittest.main()
