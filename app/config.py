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
        self.input = Size(3, 224, 224)
