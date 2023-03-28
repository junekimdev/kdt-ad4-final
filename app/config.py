from enum import Enum


class Size:
    def __init__(self, c: int, h: int, w: int) -> None:
        self.c = c
        self.h = h
        self.w = w


class Mode(Enum):
    TRAIN = 0
    EVAL = 1
    TEST = 2

    def str(self):
        if self == Mode.TRAIN:
            return "train"
        elif self == Mode.EVAL:
            return "eval"
        else:
            return "test"


class Config:
    def __init__(self) -> None:
        self.image_dir_name = "images"
        self.image_list_txt_name = "all.txt"
        self.batch_train = 16
        self.batch_eval = 1
        self.input = Size(3, 1208, 1920)
        self.output = self.input
        self.init_feature = 64
        self.K = 16
        self.conv_kernel = 3
        self.conv_stride = 1
        self.conv_padding = self._get_padding(self.conv_kernel)
        self.conv_bias = False
        self.conv_dropout_p = .1
        self.pool_kernel = 2
        self.pool_stride = 2
        self.upsample_kernel = 2
        self.upsample_stride = 2

    def _get_padding(self, kernel: int) -> int:
        assert kernel % 2, "kernel is not an odd number"
        return (kernel-1)//2
