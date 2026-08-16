"""Detection module for the AI YOLO detector application."""

from logging import getLogger
from pathlib import Path
from pprint import pprint

from ultralytics import YOLO


def detect(arguments) -> None:
    """Run predictions and save annotated images."""
    model_path = Path(arguments.model).resolve()
    source_path = Path(arguments.source).resolve()
    output_path = Path(arguments.output_folder).resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found: {model_path}")

    if not source_path.exists():
        raise FileNotFoundError(f"Detection source not found: {source_path}")

    # Load the trained YOLO model from the specified path.
    _logger.info(f"LOADING TRAINED MODEL: {model_path}")
    model = YOLO(str(model_path))

    # Run predictions on the source images and save the results.
    _logger.info(f"DETECTING OBJECTS IN: {source_path}")
    results = model.predict(
        save=True,
        save_txt=True,
        save_conf=True,
        device=arguments.device,
        conf=arguments.confidence,
        source=str(source_path),
        project=str(output_path),
        name=arguments.output_name)

    # Print the predictions results to the console.
    _logger.info(f"PREDICTIONS COMPLETED.")
    print("Results:")
    pprint(results)


_logger = getLogger(__name__)
