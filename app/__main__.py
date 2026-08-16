"""Main entrypoint for the AI YOLO detection application."""

import argparse
from logging import getLogger

from app.detect import detect
from app.train import train


DEFAULT_BASE_MODEL = "yolo26n.pt"
DEFAULT_DATASET_FOLDER = "./dataset"
DEFAULT_OUTPUT_FOLDER = "./output"
DEFAULT_OUTPUT_NAME = "train"
DEFAULT_TRAINED_MODEL = f"{DEFAULT_OUTPUT_FOLDER}/{DEFAULT_OUTPUT_NAME}/weights/best.pt"


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train or run the YOLO detector.")
    commands = parser.add_subparsers(dest="command", required=True)

    # Set up the command-line arguments for the "train" command.
    train_parser = commands.add_parser("train", help="Train a custom detector.")

    # Folder parameters.
    train_parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET_FOLDER,
                              help="Input dataset folder.")
    train_parser.add_argument("-o", "--output-folder", default=DEFAULT_OUTPUT_FOLDER,
                              help="Output folder for training results.")
    train_parser.add_argument("-n", "--output-name", default=DEFAULT_OUTPUT_NAME,
                              help="Output subfolder for this training.")
    train_parser.add_argument("-m", "--model", default=DEFAULT_BASE_MODEL,
                              help="Model to adapt to the dataset.")

    # Data parameters.
    train_parser.add_argument("-b", "--batch", type=int, default=8,
                              help="Number of images processed together per step.")
    train_parser.add_argument("-e", "--epochs", type=int, default=100,
                              help="Number of complete passes over the training images.")
    train_parser.add_argument("-s", "--image-size", type=int, default=640,
                              help="Internal image size used during training.")

    # Computing parameters.
    train_parser.add_argument("-d", "--device", default="0",
                              help='Compute device: # for the GPU or "cpu" for the CPU.')
    train_parser.add_argument("-w", "--workers", type=int, default=0,
                              help="Number of worker processes used to load images.")

    # Set up the command-line arguments for the "detect" command.
    detect_parser = commands.add_parser("detect", help="Detect objects in images.")
    # Folder parameters.
    detect_parser.add_argument("-s", "--source", required=True,
                               help="Input image, folder, or video.")
    detect_parser.add_argument("-o", "--output-folder", default=DEFAULT_OUTPUT_FOLDER,
                               help="Output folder for detection results.")
    detect_parser.add_argument("-n", "--output-name", default="detect",
                               help="Output subfolder for this detection.")
    # Model parameters.
    detect_parser.add_argument("-m", "--model", default=DEFAULT_TRAINED_MODEL,
                               help="Trained .pt model used for detection.")

    # Computing parameters.
    detect_parser.add_argument("-d", "--device", default="0",
                              help='Compute device: # for the GPU or "cpu" for the CPU.')
    detect_parser.add_argument("-c", "--confidence", type=float, default=0.25,
                               help="Minimum confidence required to keep a detection.")

    return parser.parse_args()


def main(arguments):
    """Call the appropriate function based on the arguments."""
    _logger.info(f"RUNNING COMMAND: {arguments.command.upper()}")
    try:
        if arguments.command == "train":
            train(arguments)
        else:
            detect(arguments)
    except Exception as e:
        _logger.error(f"ERROR DURING {arguments.command.upper()}: {e}")


_logger = getLogger(__name__)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
