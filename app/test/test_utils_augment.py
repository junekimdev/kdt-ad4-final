import unittest
import albumentations as A
from utils.augment import MyTransform
from app.config import Mode


class MyTransformTestCase(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tf_train = MyTransform(Mode.TRAIN)
        self.tf_eval = MyTransform(Mode.EVAL)
        self.tf_test = MyTransform(Mode.TEST)

    def test_1_init(self):
        self.assertIs(self.tf_train.mode, Mode.TRAIN)
        self.assertIs(self.tf_eval.mode, Mode.EVAL)
        self.assertIs(self.tf_test.mode, Mode.TEST)

    def test_2_get_tf_train(self):
        self.assertIsInstance(self.tf_train._get_tf_train(), A.Compose)
        self.assertIsInstance(self.tf_eval._get_tf_train(), A.Compose)
        self.assertIsInstance(self.tf_test._get_tf_train(), A.Compose)

    def test_3_get_tf_eval(self):
        self.assertIsInstance(self.tf_train._get_tf_eval(), A.Compose)
        self.assertIsInstance(self.tf_eval._get_tf_eval(), A.Compose)
        self.assertIsInstance(self.tf_test._get_tf_eval(), A.Compose)

    def test_4_get(self):
        self.assertIsInstance(self.tf_train.get(), A.Compose)
        self.assertIsInstance(self.tf_eval.get(), A.Compose)
        self.assertIsInstance(self.tf_test.get(), A.Compose)


if __name__ == '__main__':
    unittest.main()
