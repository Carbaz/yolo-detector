"""Training module for the AI YOLO detector application."""

from logging import getLogger
from pathlib import Path
from pprint import pprint

from ultralytics import YOLO
from ultralytics.utils import SETTINGS


def train(arguments) -> None:
    """Train a detector using the configured dataset."""
    # Resolve project paths before passing them to Ultralytics.
    output_path = Path(arguments.output_folder).resolve()
    dataset_path = Path(arguments.dataset).resolve()
    data_path = dataset_path / "data.yaml"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found: {data_path}")

    # Bare model names use Ultralytics' shared weights cache; explicit paths stay local.
    model_path = Path(arguments.model)
    if not model_path.is_absolute() and len(model_path.parts) == 1:
        model_path = Path(SETTINGS["weights_dir"]) / model_path
    else:
        model_path = model_path.resolve()

    # Load the base YOLO model from the specified path.
    _logger.info(f"LOADING MODEL: {model_path}")
    model = YOLO(model_path)

    # Train the model using the dataset.
    _logger.info(f"TRAINING WITH DATASET: {dataset_path}")
    results = model.train(
        data=str(data_path),
        batch=arguments.batch,
        epochs=arguments.epochs,
        device=arguments.device,
        workers=arguments.workers,
        imgsz=arguments.image_size,
        project=str(output_path),
        name=arguments.output_name)

    # Print the detection results to the console.
    _logger.info(f"TRAINING COMPLETED.")
    print("Results:")
    pprint(results)


_logger = getLogger(__name__)
