import argparse
import sys
import os
import numpy as np
from torch import Tensor
from PIL import Image
from app.config import Config
config = Config()


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Deep Learning Image Segmentation with [W-NET]")
    parser.add_argument(
        "--mode", type=str, dest="mode", default="train",
        help="one of [train | eval | test | onnx]")
    parser.add_argument(
        "-d", "--dataset", type=str, dest="dataset_root", default="./dataset",
        help="give dataset root directory (input)")
    parser.add_argument(
        "-o", "--out", type=str, dest="output_dir", default="./output",
        help="set output directory")
    parser.add_argument(
        "-c", "--checkpoint", type=str, dest="checkpoint",
        help="filename of the saved trained model")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()


def save_image(inference: Tensor, output_dir: str, filename: str):
    # change shape to h,w,3
    img_hwc = inference.squeeze(0).permute(1, 2, 0)
    assert img_hwc.shape[-1] == 3, "Tensor's channel is not 3"

    # convert to numpy array
    img_np = np.array(img_hwc.tolist(), dtype=np.uint8)
    img_res = Image.fromarray(img_np)  # convert to pillow image

    dname = os.path.join(output_dir, config.image_dir_name)
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass

    fname = os.path.join(dname, f"{filename}.jpg")
    img_res.save(fname)
    print(f"An image has been saved as {fname}")


def save_clusters(inference: Tensor, k: int, output_dir: str, filename: str):
    dname = os.path.join(output_dir, config.image_dir_name)
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass

    # change shape to h,w,3
    img_k = inference.squeeze(0)

    for i in range(k):

        # convert to numpy array
        img_np = np.array(img_k[i].tolist(), dtype=np.uint8)
        img_res = Image.fromarray(img_np)  # convert to pillow image

        fname = os.path.join(dname, f"{filename}-{i}.jpg")
        img_res.save(fname)
        print(f"An image has been saved as {fname}")
