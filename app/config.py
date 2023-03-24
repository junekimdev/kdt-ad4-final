class Size:
    def __init__(self, c: int, h: int, w: int) -> None:
        self.c = c
        self.h = h
        self.w = w


class Config:
    def __init__(self) -> None:
        self.input = Size(3, 224, 224)
