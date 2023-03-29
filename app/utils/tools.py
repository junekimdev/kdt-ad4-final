import argparse
import sys


def arg_parse():
    parser = argparse.ArgumentParser(
        description="Deep Learning Image Segmentation with [W-NET]")
    parser.add_argument(
        "-m", "--mode", type=str, dest="mode", default="train",
        help="one of [train | eval | test | onnx]")
    parser.add_argument(
        "-d", "--dataset", type=str, dest="dataset_root", default="dataset",
        help="give dataset root directory (input)")
    parser.add_argument(
        "-o", "--out", type=str, dest="output_dir", default="output",
        help="set output directory")
    parser.add_argument(
        "-c", "--checkpoint", type=str, dest="checkpoint",
        help="filename of the saved trained model")

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()
