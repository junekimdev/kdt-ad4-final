import albumentations as A
from albumentations.pytorch import ToTensorV2
from app.config import Config, Mode
config = Config()


class MyTransform:
    def __init__(self, mode: Mode) -> None:
        self.mode = mode

    def get(self):
        return self._get_tf_train() if self.mode is Mode.TRAIN else self._get_tf_eval()

    def _get_tf_train(self):
        return A.Compose([
            A.Resize(config.input.h, config.input.w,
                     always_apply=True),  # required
            A.OneOf([
                A.RandomRain(),
                A.RandomFog(),
                A.RandomSunFlare(),
                A.RandomSnow(),
                A.RandomContrast(),
                A.RandomBrightness(),
            ], p=.2),
            A.GaussianBlur(p=.7),
            A.GaussNoise(),
            ToTensorV2()  # required
        ])

    def _get_tf_eval(self):
        return A.Compose([
            A.Resize(config.input.h, config.input.w,
                     always_apply=True),  # required
            ToTensorV2()  # required
        ])
